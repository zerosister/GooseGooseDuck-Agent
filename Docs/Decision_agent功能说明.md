# DecisionAgent 功能说明

## 1. 定位与职责

`DecisionAgent`（`work_B/backend/agents/decision_agent.py`）是面向「鹅鸭杀」类社交推理游戏的**会议阶段辅助决策**组件。它在单次调用中完成两件事：

1. **前序发言分析**：结合对局上下文、记忆与（可选）最近发言，分析已发言玩家的立场与逻辑。
2. **发言建议**：为**当前被辅助的玩家**生成一段可照读的口语化发言建议。

规则类信息默认经 **`RagSummarizeService`**：先由 **`RuleLibrary`** 向量检索，再经 `prompts/rag_summarize.txt` 驱动的 **LLM 摘要**，将「规则摘要」写入提示词中的 **RAG_EXCERPTS**；也可通过配置关闭摘要，仅注入向量库**原文摘录**。

---

## 2. 依赖与数据流

| 组件 | 作用 |
|------|------|
| `RagSummarizeService` | 封装规则库检索与 RAG 摘要链（`arag_summarize`）；内部持有 `RuleLibrary` 与 `Retriever`。 |
| `chat_model`（`model.factory`） | 执行最终决策的聊天模型（与 RAG 摘要链所用模型独立配置于同一 factory）。 |
| `load_decision_prompt()` | 从配置读取决策提示词模板（默认见 `config/prompts.yaml` → `prompts/decision_agent.txt`）。 |
| `agent_conf["decision_agent"]` | 行为参数（见下文「配置」）。 |

**输入：**

- `DecisionContext`：会话 ID、当前玩家座位/ID、角色名、阵营、各轮会议状态、**玩家 roster**（颜色/座位/ID 对应）等。
- `MemorySummary`：上游记忆 Agent 产出的结构化摘要。
- `ingestions`：可选的近期「读入」记录（如语音识别等），会序列化为 JSON 一并交给模型。

**输出：**

- `DecisionResult`：`prior_speech_analysis`、`speech_suggestion`、`rag_queries_used`、`warnings` 等。

---

## 3. 规则检索（RAG）策略

### 3.1 查询构造 `_build_rag_queries`

根据 `DecisionContext` 自动生成最多 `max_rag_queries` 条检索语句（默认 2，见配置）：

- 始终包含一条：**角色技能与规则要点**（`鹅鸭杀 {role_name} 角色技能与规则要点`）。
- 若 `alignment` 为非空字符串且不为 `unknown`，追加一条：**阵营胜利条件与基本玩法**（`鹅鸭杀 {alignment} 阵营 …`）。
- 去重后截断到 `max_rag_queries`。

### 3.2 摘要模式与原文模式

由 `use_rag_summarize` 控制（默认 `true`）：

- **`true`**：对每条查询调用 `RagSummarizeService.arag_summarize(query)`，将「查询 + 规则摘要」拼入 `rag_excerpts`。
- **`false`**：对每条查询用 `Retriever` 拉取文档（`ainvoke` / 线程回退），格式化为「查询 + 原文摘录」（含 `page_content` 与 `metadata`）；无命中时标明「无命中」。

异常时记录日志，`warnings` 中追加说明，对应块内标注「失败」。

### 3.3 近期读入截断

`recent_ingestion_n`（默认见 `agent.yaml`）控制注入模型的最近 `IngestionOutput` 条数：若超过则只保留列表末尾 `recent_n` 条，再 `json.dumps` 序列化。

---

## 4. 提示词与输出解析

### 4.1 提示词要点（`prompts/decision_agent.txt`）

- 要求模型严格使用 **player_roster** 做玩家标识，不编造未出现的信息。
- 规则结论须能在「规则库检索摘录」中找到依据；否则应明确说明未检索到相关条目。
- 输出**仅**两段正文，且必须包含固定标记行：  
  `【前序发言分析】` 与 `【发言建议】`（含方括号）。

### 4.2 解析逻辑 `_parse_marked_output`

- 若两段标记均存在：按标记切分，前者进入 `prior_speech_analysis`，后者进入 `speech_suggestion`。
- 若缺少标记或格式不符：将全文归入分析段，`speech_suggestion` 置空，并在 `warnings` 中说明「未按标记格式输出」等。

---

## 5. 配置项（`config/agent.yaml`）

在 `decision_agent` 节点下：

| 键 | 含义 |
|----|------|
| `recent_ingestion_n` | 注入提示词的最近读入条数上限；为 **0** 时不截断，将全部 `ingestions` 序列化注入（列表仍可为空）。 |
| `max_rag_queries` | 自动构造并执行的规则库检索查询条数上限。 |
| `use_rag_summarize` | 为 `true` 时使用 RAG 摘要；为 `false` 时仅用向量检索原文摘录。 |

提示词路径在 `config/prompts.yaml` 的 `decision_agent_prompt_path` 中配置。

---

## 6. 与其它模块的关系

- **记忆**：消费 `MemorySummary`，不负责写入长期/短期记忆。
- **规则库**：`RagSummarizeService` 内部使用 `RuleLibrary`；索引构建与增量入库由 `RuleLibrary.load_document()` 独立完成（通常在维护脚本或单独流程中调用）。
- **图编排**：可被 `my_graph` 等 LangGraph 工作流节点调用，作为决策步骤的实现；可注入自定义 `RagSummarizeService` 实例以便测试或共享连接。

---

## 7. 本地调试

文件末尾提供 `_parse_marked_output` 的简单样例；完整端到端测试可参考 `work_B/test/run_memory_decision_graph.py` 等脚本。
