# Schema 机制
对于 LangGraph 中作为上下文保存的重要工具。当你调用 `graph.invoke(input)` 时，LangGraph 会检查你的输入字典。如果你传入了 Schema 中未定义的键，LangGraph 通常会忽略这些多余的键，只将定义的字段存入 State。通常可以定义一个继承 `TypeDict` 的类或者 `pydantic.BaseModel` 的类作为 `graphstate`

如果你想让某些字段（如日志、对话历史）不断增加而不是被覆盖，你需要使用 `Annotated` 配合一个 `reducer` 函数：
```python
from typing import Annotated
from operator import add

class AnalysisState(TypedDict):
    # 使用 add 意味着新旧列表会拼接，而不是覆盖
    speeches: Annotated[List[Dict], add]
```