import time
import re
import json
from typing import Optional, Dict
from collections import deque
from backend.utils.logger import log
import asyncio
from backend.app.core.ws_manager import ws_manager

class GGDCoordinator:
    def __init__(self, vision_service, history_limit=200):
        self.vision = vision_service
        # 视觉快照队列，保存过去一段时间的画面状态
        self.vision_history = deque(maxlen=history_limit)
        log.success("GGD 协调器：已准备好作为决策中枢运行。")
        
        # 聚合状态追踪
        self.last_speaker_id = -1  # 初始设为不存在的 ID
        self.current_session = None

    def record_vision_state(self, res):
        """由 GGDVisionService 每秒/每帧调用"""
        self.vision_history.append({
            "ts": time.time(),
            "seat": res.get("active_seat", None),
            "ui": res.get("ui_text", "")
        })

    def on_audio_result(self, text: str, start_time: float, duration: float, asr_spk: str):
        """处理 ASR 结果并根据 Speaker ID 聚合文本"""
        # 1. 判定说话人 (优先级：UI > HSV > Audio)
        speaker_info = self._determine_speaker(start_time, duration, asr_spk)
        sid = speaker_info["seat_id"]
        
        # 2. 聚合判定：如果 Speaker 未变，则在原文后追加 4 个空格和新文本
        if sid == self.last_speaker_id and self.current_session:
            self.current_session["content"] += f"    {text}"
            self.current_session["type"] = "update"
        else:
            # 说话人变更或首次说话，开启新条目
            self.current_session = {
                "type": "new",
                "id": int(time.time() * 1000),
                "timestamp": round(start_time, 2),
                "seat_id": sid,
                "name": speaker_info["name"],
                "content": text,
                "method": speaker_info["method"]
            }
        
        self.last_speaker_id = sid
        self._dispatch_result(self.current_session)

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
    
    def _dispatch_result(self, payload: dict):
        """分发结果至 WebSocket"""
        log.success(f"Final JSON Output: {payload['name']} - {payload['content'][:15]}...")
        
        # 获取当前运行的事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在正在运行的 loop 中安全地创建异步任务
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)
            else:
                log.warning("事件循环未运行，无法发送消息")
        except Exception as e:
            log.error(f"协调器推送消息异常: {e}")