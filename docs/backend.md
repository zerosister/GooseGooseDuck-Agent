- 为什么将音频处理与视频帧处理作为两个**进程**来用而非两个**线程**？
  - 因为存在`GIL`（全局解释器锁），在多线程环境下，如果某个线程占用了`GIL`，其他线程就只能等待，这会导致程序的运行效率下降。因此，为了提高程序的运行效率，我们将音频处理与视频帧处理分离成两个进程。**这样对于多核CPU，可以充分利用多核资源**，提高程序的运行效率。
- 多进程日志（`enqueue=Ture`）
  - 在`Loguru`中设置`enqueue=Ture`会将日志放入队列，专门利用一个线程进行写入，防止ASR，OCR进程实时处理因此卡顿。

# 音频处理

## ~~原始方案 v0~~

采用库 Soundcard 比 Sounddevice 好用很多。`soundcard` 内置`recorder`可以调用windows的重采样引擎。且通过共享模式读取音频缓存，不会竞争蓝牙耳机带宽。

- _silero_vad：利用模型执行音频`chunk`的静音判定
- audio_segmenter：调用 vad 模型，并根据最长静音阈值等将音频合并为小段，并将其放入跨进程的`task_queue`中，待 ASR 模型使用
- audio_capture：录制音频，创建`chunk`对象，并将音频写入单进程的`audio_queue`中

### 同步设置

在处理音频过程中，由于需要将录下的声音进行 VAD 寻找人声，会对原始的音频进行裁剪。所以需要记录每段音频开始和结束的时间戳，这样才能与后期 ASR 的结果对应起来。

### silero VAD 细节
silero VAD 的实现针对每一个`chunk`的大小都是有限制的，如果采样率为16k，则要求`chunk`的大小一定要为512，也就是32ms，所以音频本身就切分的足够小。最终需要的就是将所有判定为`True`的`chunk`合并起来。

## Sherpa onnx 方案一

单进程模式，音频收集，说话人识别与ASR顺序执行。

### 录音 [AudioRecorder](../backend/app/services/audio_service/audio_recorder.py)

采用 `soundcard` 库进行录音，使用内置的`recorder`进行回环（将电脑正在播放的声音重新捕捉并作为输入信号处理称为回环）录音。

### sherpa onnx VAD 使用

利用 `sherpa_onnx.VadModelConfig()` 配置vad模型，silero_vad=SileroVadModelConfig(model="", threshold=0.5, min_silence_duration=0.5, min_speech_duration=0.25, max_speech_duration=20, window_size=512, neg_threshold=-1)

### 说话人识别 [SpeakerEngine](../backend/app/services/audio_service/speaker_engine.py)

- 采用模型：3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx，模型大小 110 M

出现问题：`SoundcardRuntimeWarning: data discontinuity in recording
  warnings.warn("data discontinuity in recording", SoundcardRuntimeWarning)`

- 解决方案：将音频采集线程作为主线程。Speaker识别与ASR放在子线程中。采用**生产者-消费者**模型 [audio_capture](../backend/app/services/audio_service/audio_capture.py)

# 视频帧处理

OCR 需要异步单开一个线程做，因为除开座位识别，说话人识别仍然需要拾取特定区域的文字。

## ~~座位识别（已弃用，代码中有实现）~~

**preprocess_img()**
- 获得灰度图
- HE均衡化
- 二值化
- 开运算

这样操作后得到的14，15号位均为纯白色，由此可以得到相邻卡片x方向的距离，再通过预设的y方向距离比例`_GGD_ROW_RATIO`，可推算出所有的座位`roi`

## 座位识别

采用前端 ui 坐标系框选的方式，选中1,2,4号玩家的座位卡片框，以及发言检测区的UI框。发送回后端后会进行剩余的座位推导，并生成`roi_config.json`文件。

## [vision_ocr.py](../backend/app/services/vision_ocr.py)

- 玩家名称识别（OCR）
- 发言状态检测
  - 根据边框黄色判定（HSV色域方法，快）
  - 根据UI内容判定（OCR方法，慢）

扫描的 loop 为独立的，但这个 loop 主要应用于后续的说话人识别 OCR，在检测到规定人数的名称后，玩家名称识别将被跳过。

由于只用识别单行文字，所以只使用了 PP-OCRv5 的 Recongnition 内容。见[rapidcor.yaml](../rapidocr.yaml)

最后，将**结果**传递到[GGD-Coordinator](../backend/app/core/ggd_coordinator.py)中：
```json
res = {
    "active_seat": speaker_id, 
    "ui_text": self.current_ui_text,
}
```

# [ggd_coordinator.py](../backend/app/core/ggd_coordinator.py)

对音频 ASR 与视频 OCR 的结果进行整合的协调器。

- 持续收集视频帧处理的结果（见 [座位识别](#vision_ocrpy) 部分），并将其存入`vision_history`中。
- 对于音频 ASR 结果，采用触发的方式——收到了一段 ASR 结果就开始处理（耗时长），首先进行说话人判定，再将结果发送到前端。
  - 三重保险判定说话人
    - 最高优先级：UI 内容判定（发言检测区中会写“xx发言中，xx请准备”）
    - 次高优先级：HSV 对话框判定（发言者的座位卡片边框变黄色，当黄色区域超过阈值时确定发言者）。当 UI 内容无法识别出号码时，采用此方法。
    - 最低优先级：根据 audio 处理中的 speaker 识别进行判定。需要根据 vision_history 中发言者出现的结果进行判定。（但也可基本不用此方法，因为无法与座位 ID 对应）


# [main.py](../backend/main.py) 解释

- lifespan：异步上下文管理器。
  - audio_thread：创建守护进程进行音频采集，防止音频采集的阻塞操作（如等待音频信号）导致 FastAPI 的异步主线程（处理网络请求的线程）卡死。
  - `daemon=True`：守护进程能够随着主进程一同自毁。
  - yield：当程序执行到 yield 时，它会暂停在这里，并通知 FastAPI：“启动准备已完成，现在可以开始接收客户端请求了”

```python
# Vue 3 前端通常运行在不同的端口（如 5173），必须允许跨域请求，否则前端无法连接到后端的 WebSocket 或 REST 接口。

app.add_middleware(
    CORSMiddleware,   
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    
```

`app.include_router()`：将路由添加到应用中，挂载到某个路径如`/api/v1`。

`app.state` 为一个 `FastAPI()` 实例的一个存储容器，典型存储内容：数据库连接池、深度学习模型（权重文件）

`request.app.state` 是在 **路由函数** 内部访问同一个存储容器的方式，在一个 HTTP 请求中，FastAPI 会把当前的 `app` 实例注入到 `request` 对象中。因此，`request.app.state` 实际上就是你在 `main.py` 中定义的那个 `app.state` 的引用。

# [vision_router.py](../backend/app/api/vision_router.py) 解释

作为前后端交互的接口层（Router）

- vision_websocket：这是一个WebSocket（长连接）接口函数，建立一个持久的双向数据通道，专门用于处理高频的图像帧数据。
- get_service_status: 用于简单的健康检查（Health Check），告知前端后端 agent 是否在线。

```python
app.include_router(vision_router, prefix="/api/v1")
```
这意味着 vision_router 内部定义的所有路径都会自动带上这个前缀。它们的关系是父子嵌套关系，如`/api/v1/ws/stream`和`/api/v1/status`
