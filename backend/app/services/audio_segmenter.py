import numpy as np
import time
import os
import soundfile as sf
from backend.utils.logger import log
from backend.utils.config_loader import config

class AudioSegmenter:
    def __init__(self, task_queue, sample_rate=16000, save_dir="backend/tests/debug_segments"):
        self.task_queue = task_queue
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        
        # 配置参数
        self.max_silence_time = config.audio.max_silence_time  # 容忍的静音时长（Hangover Time）
        self.min_segment_time = config.audio.min_segment_time    # 送去 ASR 的最低时长
        self.max_segment_time = config.audio.max_segment_time   # 强制断句，防止过长
        
        # 内部状态
        self.active_buffer = []       # 存放当前说话段的 PCM 块
        self.start_ts = 0             # 当前段落的起始时间
        self.last_speech_time = 0     # 最后一次检测到人声的时间
        self.is_recording = False     # 是否处于“正在录制有效段落”状态

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def process_block(self, pcm_block, start_ts, end_ts, is_speech):
        """
        处理每一个传入的 block
        is_speech: 由 VAD 模块判定后的布尔值
        """

        if is_speech:
            # --- 情况 A: 发现人声 ---
            if not self.is_recording:
                # 开启新段落
                self.is_recording = True
                self.start_ts = start_ts
                log.debug(f"检测到人声，开启新段落: {self.start_ts}")
            
            self.active_buffer.append(pcm_block)
            self.last_speech_time = end_ts
            
            # 检查是否过长
            if (end_ts - self.start_ts) > self.max_segment_time:
                self._finalize_segment(reason="timeout")

        else:
            # --- 情况 B: 发现静音 ---
            if self.is_recording:
                # 检查是否超过了容忍的停顿时间
                silence_duration = end_ts - self.last_speech_time
                if silence_duration > self.max_silence_time:
                    self._finalize_segment(reason="silence")
                else:
                    # 虽然是静音 block，但在停顿容忍期内，仍暂存入 buffer
                    # 这样可以保证 ASR 识别时句子的连贯性
                    self.active_buffer.append(pcm_block)

    def _finalize_segment(self, reason="normal"):
        """完成一段音频的提取、保存与分发"""
        if not self.active_buffer:
            return

        full_pcm = np.concatenate(self.active_buffer)
        duration = len(full_pcm) / self.sample_rate

        # 长度过滤
        if duration < self.min_segment_time:
            # 长度过短，不送去 ASR
            self.active_buffer = []
            self.is_recording = False
            return

        # 保存文件用于 Debug 和 ASR
        timestamp_str = time.strftime("%H%M%S", time.localtime(self.start_ts))
        filename = f"seg_{timestamp_str}_{int(self.start_ts*1000)}.wav"
        file_path = os.path.join(self.save_dir, filename)
        
        if config.audio.save_temp:
            sf.write(file_path, full_pcm, self.sample_rate)
            log.success(f"段落结算 [{reason}]: {filename} ({duration:.2f}s)")
        else:
            log.debug(f"段落结算 [{reason}]: ({duration:.2f}s)")

        # 发送给 ASR 进程
        self.task_queue.put({
            "pcm": full_pcm,
            "start_ts": self.start_ts,
            "source": "game_audio"
        })

        # 重置状态
        self.active_buffer = []
        self.is_recording = False