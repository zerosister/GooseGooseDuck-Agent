# GooseGooseDuck-Agent

鹅鸭杀会议实时分析助手 - 多智能体系统

## 使用前提

1. 申请阿里云百炼平台 API KEY，将 `DASHSCOPE_API_KEY` 加入环境变量

## 读入 Agent（Ingestion Agent）

### 启动后端

```bash
pip install -r requirements.txt
python run_backend.py
# 或: uvicorn backend.main:app --reload --port 8000
```

### API 端点

- `GET /` - 健康检查
- `GET /health` - 健康检查
- `WS /ws/speech` - WebSocket 语音流，发送 PCM 16kHz 16bit 音频帧，接收 IngestionOutput JSON
- `POST /api/v1/ingestion/image` - 上传会议截图（multipart/form-data）
- `POST /api/v1/ingestion/image/base64` - 上传会议截图（JSON base64）

### 开发期音频采集（本地播放视频模拟）

```bash
pip install PyAudioWPatch websockets
python scripts/audio_capture_dev.py
# 在本地播放鹅鸭杀视频，脚本将捕获系统音频并发送到后端
```

### 配置

- `config/ingestion.yaml` - ASR、Vision、Emotion 参数
- `config/prompts.yaml` - 各 prompt 路径

