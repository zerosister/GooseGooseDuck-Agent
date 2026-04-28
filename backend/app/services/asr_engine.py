import multiprocessing
import numpy as np
from faster_whisper import WhisperModel
from backend.utils.logger import log
from backend.utils.config_loader import config

class ASREngine(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        """
        :param task_queue: 接收任务字典的队列 {"pcm": array, "start_ts": float, "source": str}
        :param result_queue: 发送识别结果的队列 {"text": str, "source": str, "start_ts": float}
        """
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.is_running = multiprocessing.Value('b', True)

    def run(self):
        """子进程入口：加载模型并开始消费 PCM 数据"""
        log.info(f"ASR 子进程启动，加载模型: {config.asr.model_size}")
        
        # 在子进程中初始化模型，确保显存/内存隔离
        model = WhisperModel(
            config.asr.model_size,
            device=config.asr.device,
            compute_type=config.asr.compute_type
        )
        log.success("ASR 子进程模型加载完成")

        while self.is_running.value:
            try:
                # 获取任务数据包
                task = self.task_queue.get(timeout=1.0)
                
                audio_data = task.get("pcm")
                start_ts = task.get("start_ts")
                source = task.get("source", "unknown")

                if audio_data is None or len(audio_data) == 0:
                    continue

                # 核心改动：直接传入 NumPy 数组进行识别
                # faster-whisper 要求输入为 float32 类型的平铺数组
                segments, _ = model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="zh",
                    initial_prompt="这是鹅鸭杀游戏的语音，包含：狼人、好人、中立、身份、逻辑、发言。"
                )

                # 拼接文本结果
                full_text = "".join([s.text for s in segments]).strip()

                if full_text:
                    log.success(f"ASR 识别成功 [{source}]: {full_text}")
                    
                    # 关键：将识别出的文本与原始时间戳一起发回主进程
                    # 这样主进程才知道这段话是“什么时候”说的
                    self.result_queue.put({
                        "text": full_text, 
                        "source": source, 
                        "start_ts": start_ts
                    })

            except multiprocessing.queues.Empty:
                continue
            except Exception as e:
                log.error(f"ASR 进程处理异常: {e}")
                continue

    def stop(self):
        self.is_running.value = False
        self.terminate()