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