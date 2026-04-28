import os
import time
import queue
import threading
import torch
import numpy as np
import soundfile as sf
import soundcard as sc
from backend.utils.logger import log
from backend.utils.config_loader import config
from backend.utils.pathtool import get_abs_path
from backend.app.services.audio_segmenter import AudioSegmenter
from backend.app.services._silero_vad import SileroVAD

class AudioCaptureService:
    def __init__(self, task_queue):
        # 基础配置
        self.sample_rate = 16000  # Silero VAD 强制要求 16000 或 8000
        self.save_path = get_abs_path(config.audio.save_path)
        self.task_queue = task_queue
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # VAD 配置
        self.threshold = getattr(config.audio, "vad_threshold", 0.5)  # VAD 模型推理概率
        self.vad = SileroVAD(
            model_path=get_abs_path(config.audio.vad_model_path), 
            threshold=self.threshold, 
            sample_rate=self.sample_rate)
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _get_loopback_device(self):
        """锁定系统的回环采样设备（扬声器输出）"""
        try:
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            for mic in mics:
                if getattr(mic, "isloopback", False) and mic.id == default_speaker.id:
                    log.success(f"已锁定输出回环设备: {mic.name}")
                    return mic
            return None
        except Exception as e:
            log.error(f"探测音频设备失败: {e}")
            return None

    def _save_worker(self):
        """
        核心处理逻辑：
        每隔一定时间处理一次缓冲区，使用 get_speech_timestamps 剔除空白
        """
        log.info("音频处理线程已启动 (使用 get_speech_timestamps 模式)")
        segmenter = AudioSegmenter(self.task_queue, self.sample_rate, self.save_path)
        
        while self.is_running:
            try:
                # 从队列获取原始 PCM
                packet = self.audio_queue.get(timeout=1.0)

                # 1. 执行 VAD 判定
                audio_chunk = packet["pcm"]
                is_speech = self.vad.is_speech(audio_chunk)

                # 2. 喂给 Segmenter
                segmenter.process_block(
                    packet['pcm'],
                    packet['start_ts'],
                    packet['end_ts'],
                    is_speech
                )
                    
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"处理线程异常: {e}")

    def run(self):
        """采集进程主入口"""
        loopback_mic = self._get_loopback_device()
        if not loopback_mic:
            log.error("无法找到回环设备，采集进程退出")
            return

        self.is_running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # 采样块大小
        block_size = 512
        
        log.info("开始实时音频采集...(Block Size: {block_size})")
        try:
            with loopback_mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while self.is_running:
                    # 获取数据，阻塞的方法，会一直录直到填满 block_size
                    data = recorder.record(numframes=block_size)

                    # 记录当前系统时间，作为这一块音频的结束时间
                    current_ts = time.time()

                    # 计算音频开始时间
                    duration = len(data) / self.sample_rate
                    start_ts = current_ts - duration
                    
                    # 混音为单声道
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    
                    # 转换为 float32 并归一化
                    audio_chunk = data.astype(np.float32)

                    # 将包含时间戳的字典包传入队列
                    packet = {
                        "pcm": audio_chunk,
                        "start_ts": start_ts,
                        "end_ts": current_ts,
                        "sample_rate": self.sample_rate
                    }
                    self.audio_queue.put(packet)
                    
        except Exception as e:
            log.error(f"采集循环异常: {e}")
        finally:
            self.is_running = False
            log.info("采集进程已停止")

    def stop(self):
        self.is_running = False