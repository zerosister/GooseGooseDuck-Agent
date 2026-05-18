# 前端对后端 API 的使用

接收 webSocket 的数据类型：

```json
{
    "type": "new", // process_frame 或 new 或 update
    "id": int(time.time() * 1000),
    "timestamp": round(item["start_time"], 2),
    "seat_id": sid,
    "name": speaker_info["name"],
    "content": item["text"],
    "method": speaker_info["method"]
}
```

- `type`：表示数据类型，`new` 表示 ggd-coordinator 发送的新发言人数据，`update` 表示更新发言人数据，`process_frame` 表示处理视频帧数据后得到的**说话人ID**。

# ~~视频帧抓取（弃用）~~

在最初版本中，选择在浏览器端进行视频帧抓取，出现两个问题：
1. 每次刷新前端网页，需要重新选择窗口监听
2. 经常监听着监听着，网页上的画面就不动了。

这是由浏览器的**安全机制**（Security Sandboxing）决定的。getDisplayMedia 接口要求每次调用必须由“用户手势”（如点击按钮）触发，且出于隐私考虑，**浏览器不会持久化录屏权限**。

如果你的前端标签页失去焦点（切换到了游戏窗口），浏览器为了**省电**会降低甚至停止计时器（setInterval）和渲染请求。

所以采用在后端抓取视频帧的方式，可以解决以上两个问题。
前端只负责展现即可。

# 核心架构
[App.vue](../frontend/src/App.vue) 是前端总控组件，负责：
- 建立 WebSocket：`/api/v1/ws/stream`
- 根据消息类型分发给视觉面板和语音日志
- 控制三个面板显示：视觉、日志、设置
- 控制 Electron 窗口鼠标穿透和大小调整

子- 组件通过 ref + defineExpose 暴露方法给父组件调用，比如 `ScreenCapture.handleVisionData()`、`SpeechLog.handleIncomingSpeech()`。
`provide('ws_context', ...)` 把 WebSocket 发送能力下发给组件，[ScreenCapture.vue](../frontend/src/components/ScreenCapture.vue) 用它发送 `request_capture_frame`。

## 三个重要组件
- [ScreenCapture.vue](../frontend/src/components/ScreenCapture.vue) ：视觉面板。用 canvas 显示后端传来的图像帧，支持缩放、显示 ROI 网格、拖拽标定座位区域，并通过 /api/v1/calibrate 保存标定。
- [SpeechLog.vue](../frontend/src/components/SpeechLog.vue)：实时语音日志。维护 logs 列表，处理 new/update/processed_frame，显示当前视觉锁定的发言座位。
- [SettingsPanel.vue](../frontend/src/components/SettingsPanel.vue)：设置面板。读取和保存后端配置，包括 host、port、采集模式、目标地址、FPS 限制。