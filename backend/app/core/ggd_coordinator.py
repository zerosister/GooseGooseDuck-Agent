import time
import re
import json
from typing import Optional, Dict
from collections import deque
from backend.utils.logger import log
import asyncio
from backend.app.core.ws_manager import ws_manager
from concurrent.futures import ThreadPoolExecutor

class GGDCoordinator:
    def __init__(self, vision_service, loop=None, history_limit=200):
        self.vision = vision_service
        # 视觉快照队列，保存过去一段时间的画面状态
        self.vision_history = deque(maxlen=history_limit)
        log.success("GGD 协调器：已准备好作为决策中枢运行。")
        
        # 聚合状态追踪
        self.last_speaker_id = -1  # 初始设为不存在的 ID
        self.current_session = None

        # 为了不阻碍 audio_capture 线程，构建一个任务队列
        # 保存主事件循环引用
        self.loop = loop or asyncio.get_event_loop()
        self.task_queue = asyncio.Queue()

        # 确立最大线程数目
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 开启聚合任务线程
        self.worker_task = self.loop.create_task(self.process_task())

    def record_vision_state(self, res):
        """由 GGDVisionService 每秒/每帧调用"""
        self.vision_history.append({
            "ts": time.time(),
            "seat": res.get("active_seat", None),
            "ui": res.get("ui_text", "")
        })

    def on_audio_result(self, text: str, start_time: float, duration: float, asr_spk: str):
        """仅仅是把任务塞进异步队列，极速返回，不阻塞音频线程"""
        data = {
            "text": text,
            "start_time": start_time,
            "duration": duration,
            "asr_spk": asr_spk
        }
        self.loop.call_soon_threadsafe(
            self.task_queue.put_nowait, 
            data
        )
    
    async def process_task(self):
        """处理 ASR 结果并根据 Speaker ID 聚合文本"""
        log.info("GGD 协调器启动说话人聚合任务...")
        while True:
            # 等待队列中的新结果
            item = await self.task_queue.get()

            try:
                # 1. 判定说话人 (优先级：UI > HSV > Audio)
                speaker_info = await self.loop.run_in_executor(
                    self.executor,
                    self._determine_speaker,
                    item["start_time"],
                    item["duration"],
                    item["asr_spk"]
                )
                sid = speaker_info["seat_id"]
                
                # 2. 聚合判定：如果 Speaker 未变，则在原文后追加 4 个空格和新文本
                if sid == self.last_speaker_id and self.current_session:
                    self.current_session["content"] += f"    {item['text']}"
                    self.current_session["type"] = "update"
                else:
                    # 说话人变更或首次说话，开启新条目
                    self.current_session = {
                        "type": "new",
                        "id": int(time.time() * 1000),
                        "timestamp": round(item["start_time"], 2),
                        "seat_id": sid,
                        "name": speaker_info["name"],
                        "content": item["text"],
                        "method": speaker_info["method"]
                    }
                
                # 3. 发送聚合结果
                await self._dispatch_result(self.current_session)
            except Exception as e:
                log.error(f"GGD 协调器处理任务异常: {e}")
            finally:
                # 4. 标记完成
                self.task_queue.task_done()

    def _determine_speaker(self, start_ts: float, duration: float, asr_spk: str) -> Dict:
        """
        核心优先级决策：UI > HSV > Audio
        """
        end_ts = start_ts + duration
        # 提取音频区间内的视觉快照
        snaps = [s for s in self.vision_history if start_ts - 0.2 <= s["ts"] <= end_ts + 0.2]
        
        # --- P0: UI 识别 ---
        for s in reversed(snaps):
            match = re.search(r"(\d+)发言", s["ui"])
            if match:
                seat_id = int(match.group(1))
                return {"seat_id": seat_id, "name": self.vision.seat_names.get(seat_id, "Unknown"), "method": "ui_ocr"}

        # --- P1: HSV 颜色 ---
        seat_votes = {}
        for s in snaps:
            if s["seat"]:
                seat_votes[s["seat"]] = seat_votes.get(s["seat"], 0) + 1
        
        if seat_votes:
            top_seat = max(seat_votes, key=seat_votes.get)
            return {"seat_id": top_seat, "name": self.vision.seat_names.get(top_seat, "Unknown"), "method": "hsv_vision"}

        # --- P2: Audio Fallback ---
        return {"seat_id": None, "name": f"Unknown({asr_spk})", "method": "audio_engine"}
    
    async def _dispatch_result(self, payload: dict):
        """分发结果至 WebSocket（异步）"""
        log.success(f"{payload['seat_id']} -{payload['name']} 发言：{payload['content'][:15]}...")
        try:
            await ws_manager.broadcast(payload)
        except Exception as e:
            log.error(f"WS推送消息异常: {e}")

    async def stop(self):
        """停止协调器"""
        log.info("GGD 协调器停止运行...")

        # 1. 取消异步的 Worker 任务
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                log.info("GGD 协调器已停止运行。")
        
        # 2. 关闭线程池
        self.executor.shutdown(wait=True)
        log.success("GGD 协调器线程池已关闭")