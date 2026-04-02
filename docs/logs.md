在调试过程中，由于采用了将所有信息，事无巨细地记录，导致日志文件庞大，各种信息混杂在一起。
- 日志 `handler`
> `handler` 决定了消息的输出方式，可以是输出到控制台，文件，邮件等。可以进行**二次过滤**的效果，调控日志格式，进行日志生命周期管理等。
- 层级日志机制
```python
get_logger("ggd")  # 父节点
get_logger("ggd.agent")  # 子节点
```
> python log 层级结构中，子 Logger 默认会把消息传给父 Logger.如果你在每一层都添加 Handler，就会出现当你调用 `ggd.engine.info("Hello")` 时，`ggd.engine` 的 `Handler` 会写一次文件，然后它把消息传给 `ggd`，`ggd` 的 `Handler` 又会写一次文件。

那么当我想 Debug 某个子模块时，只需要设置父节点 `ggd` 的 `level` 比 `INFO` 更高（如 `ERROR`），单独设置子模块 `ggd.agent` 的 level 为 `INFO` 即可。