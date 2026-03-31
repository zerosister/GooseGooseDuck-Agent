## Async 用法

对于模型的流式输出：
```python
async for chunk in self.model.astream(prompt):
    delta = getattr(chunk, "content", None)
    if delta:
        yield delta
```
- `yield`：将一个函数变成 生成器 (Generator)。