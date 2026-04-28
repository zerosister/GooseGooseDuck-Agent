# 鹅鸭杀分析助手

准备环境：
- Windows 11 家庭版
- Python 3.11

目录结构：
```
goosegooseduck-agent/
├── backend/
│   ├── app/
│   │   ├── api/              # WebSocket 与 REST 路由
│   │   ├── core/             # 核心逻辑：多进程管理、配置
│   │   ├── services/         # 感知服务
│   │   │   ├── audio_capture.py # WASAPI Loopback 逻辑
│   │   │   ├── asr_engine.py    # Faster-Whisper CPU 封装
│   │   │   ├── speaker_id.py    # 声纹识别逻辑
│   │   │   └── vision_ocr.py    # PaddleOCR 窗口识别
│   │   └── scripts/            # 信号处理、图像预处理工具
│   ├── models/               # 存放轻量化模型 (Whisper-tiny, OCR-light)
│   ├── main.py               # 入口文件
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # 视频预览窗、文本瀑布流组件
│   │   ├── composables/      # WebRTC 捕获逻辑、WS 封装
│   │   └── App.vue
│   ├── package.json
│   └── vite.config.ts
└── utils/                  # 环境配置与日志系统
```

## 输入处理

**技术栈**
后端：
- Web框架：FastAPI
- 音频采集： 
  - soundcard（WASAPI Loopback）
  - Silero VAD（语音活动检测，有声音才进行录制）
- 实时音轨分析：
  - ASR（语音转文字）：Faster-Whisper，设备：CPU
  - 声纹识别：Wespeaker，设备：CPU
- OCR（图像文字识别）处理
  - 图像预处理：OpenCV
  - OCR：CRNN，设备：CPU
前端：
- 框架：Vue3（Vite）
- 窗口捕获：WebRTC
- 实时通信：WebSocket

### 系统架构设计

生产者-消费者模型

1. 音频进程：通过`Soundcard`库持续监听 WASAPI 流,使用跨进程队列：`multiprocessing.Queue`进行生产。
2. ASR 进程：将ASR结果放入一个`multiprocessing.Queue`队列中，即`result_queue`;

