# Audio
双线程监听
- microphone（麦克风声音）
- loopback（系统声音）

## Threading 相关
- `threading.Event()`，线程间通信机制，内部维护一个“标志位”，通过`set()`方法设置标志位，通过`clear()`方法清除标志位。[capture_service.py](../app/audio/capture_service.py)中使用该机制控制监听时间段。
- `Daemon Thread`，守护线程。它的生命周期取决于主程序。当主程序（所有非守护线程）退出时，守护线程会被 Python 强制直接杀掉，无论它是否运行完毕。两个音频流的监听均为守护线程。