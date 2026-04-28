import onnxruntime as ort
import librosa
import numpy as np
from backend.utils.logger import log

class SileroVAD:
    def __init__(self, model_path, threshold=0.5, sample_rate=16000):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold
        self.sample_rate = np.array([sample_rate], dtype=np.int64)
        
        # --- 关键修复：根据官方源码，维度应为 128 ---
        self._state = np.zeros((2, 1, 128)).astype('float32')
        
        # 不同的块大小对应不同的毫秒数 (16kHz 下，512 samples = 32ms)
        self.chunk_size = 512

        # 需要维护一个 context
        self.context_size = 64 if sample_rate == 16000 else 32
        self._context = np.zeros((1, self.context_size)).astype('float32')

    def is_speech(self, chunk):
        """
        输入 chunk: np.ndarray (float32), 长度需为 512(16KHz)
        返回: bool (True 为有人说话, False 为静音)
        """
        try:
            # 1. 维度转换与数据类型对齐
            if chunk.ndim == 1:
                chunk = chunk[np.newaxis, :]
            chunk = chunk.astype('float32')

            # 2. 拼接上下文 (Context)
            # 模型不仅看当前块，还要结合前 64 个采样点的信息
            input_combined = np.concatenate([self._context, chunk], axis=1)

            # 3. 执行 ONNX 推理
            ort_inputs = {
                'input': input_combined,
                'state': self._state,
                'sr': self.sample_rate
            }
            out, next_state = self.session.run(None, ort_inputs)

            # 4. 更新内部状态
            # 更新 RNN 的 Hidden State
            self._state = next_state
            # 更新滑动窗口的上下文 (取当前输入的最末尾部分)
            self._context = input_combined[:, -self.context_size:]

            # 5. 返回概率判定
            prob = out[0][0]
        except Exception as e:
            log.error(f"silero_vad 推理失败: {e}")
            raise e
        
        return prob > self.threshold
    
    def process_wav(self, wav_path):
        # 2. 使用 librosa 加载音频
        # sr=None 表示保持原采样率，但 Silero 强制要求 8000 或 16000
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        
        # 修改后 (推荐使用 .item())
        print(f"音频长度: {len(audio) / self.sample_rate.item():.2f} 秒")

        # 3. 循环分块检测
        speech_timestamps = []
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i : i + self.chunk_size]

            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            
            if (self.is_speech(chunk)):
                speech_timestamps.append(i / self.sample_rate.item())

        return speech_timestamps

# 使用示例
# vad = SileroVAD("D:\\Master_Phase\\LLM\\GooseGooseDuck-Agent\\backend\\models\\silero_vad.onnx")
# sts = vad.process_wav(wav_path='D:\\Master_Phase\\LLM\\GooseGooseDuck-Agent\\backend\\tests\\en.wav')
# print(sts)
