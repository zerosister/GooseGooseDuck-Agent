# GooseGooseDuck 分析助手 Baseline

## 技术栈
- API: FastAPI + Uvicorn + WebSocket
- 输入处理: RapidOCR（截图文本识别）+ FunASR（音频识别）
- 工作流: LangGraph
- LLM: DashScope ChatTongyi（默认 `qwen-turbo`）

## 项目结构
- `app/main.py`: 后端入口与 WebSocket 协议
- `app/pipeline/input_processing.py`: OCR/ASR 输入处理
- `app/pipeline/analysis_graph.py`: LangGraph 分析流
- `app/llm/tongyi_client.py`: Tongyi 调用封装
- `utils/logger.py`: 统一日志系统
- `web/index.html` + `web/app.js`: 最简前端采集页面
- `logs/`: 运行时日志目录（自动创建）

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```
ASR 模块需要下载[ffmpeg](https://www.ffmpeg.org/download.html)


2. 配置环境变量（`.env`）
```env
DASHSCOPE_API_KEY=your_api_key
```

3. 启动后端
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. 启动前端静态服务（在 `web/` 目录）
```bash
python -m http.server 5500
```

5. 打开浏览器访问 `http://127.0.0.1:5500`
   - 点击“选择窗口”
   - 点击“连接WS”
   - 点击“开始采集”
   - 点击“立即分析”

## WebSocket 消息协议
- 客户端 -> 服务端
  - `{"type":"frame","image_base64":"..."}`
  - `{"type":"audio","audio_base64":"...","mime_type":"audio/webm"}`
  - `{"type":"analyze"}`
  - `{"type":"reset"}`
- 服务端 -> 客户端
  - `{"type":"session","session_id":"..."}`
  - `{"type":"analysis_result","session_id":"...","speech_count":N,"result":{...}}`

## 日志说明
日志系统会将每个环节信息写入 `logs/`：
- `logs/app.log`: 全局运行日志
- `logs/sessions/<session_id>.jsonl`: 单会话全量事件日志
- `logs/daily/<YYYY-MM-DD>.jsonl`: 按天聚合日志
- `logs/*.png`, `logs/*.webm`: 原始截图和音频块

关键事件包括：
- WebSocket 收发、连接断开、异常
- OCR/ASR 处理结果与异常
- LangGraph 节点执行状态
- Tongyi 请求、响应、错误

## 已知限制（Baseline）
- 浏览器无法稳定实现“只捕获某游戏窗口音频”，当前基于 `getDisplayMedia` 权限能力。
- 音频格式由浏览器输出为 `audio/webm`，在部分环境下 FunASR 可能无法直接解码，会保留原始音频并记录错误日志。