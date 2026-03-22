# RuleLibrary（规则库）功能说明

## 1. 定位与职责

`RuleLibrary`（`work_B/backend/services/rag/rule_library.py`）负责**鹅鸭杀规则知识**的：

1. **持久化向量存储**：使用 Chroma + 项目统一的 **embedding 模型** 建立可检索的文档集合。
2. **增量入库**：从配置目录扫描允许类型的文件，按文件 **MD5** 判断是否已入库，仅对新文件或内容变更后的文件执行加载与分块写入。
3. **检索出口**：对外提供 LangChain `Retriever`（`search_kwargs.k` 由配置指定），供 `RagSummarizeService` 等模块做规则检索。

对检索结果再做一层 **LLM 总结**由 `RagSummarizeService`（`rag_service.py`）完成；`RuleLibrary` 本身只负责向量库与检索。

---

## 2. 技术栈

| 项目 | 说明 |
|------|------|
| 向量库 | `langchain_chroma.Chroma` |
| 嵌入 | `model.factory.embedding_model` |
| 分块 | `RecursiveCharacterTextSplitter`（块大小、重叠、分隔符来自配置） |
| 文档加载 | `utils.file_handler`：`pdf_loader`、`text_loader`、`xlsx_loader` |

---

## 3. 配置（`config/chroma.yaml` → `rule_library`）

| 键 | 含义 |
|----|------|
| `collection_name` | Chroma 集合名（如 `rule_library`）。 |
| `persist_directory` | 向量库持久化目录（相对路径经 `get_abs_path` 解析）。 |
| `k` | `as_retriever(search_kwargs={"k": k})` 每次检索返回的文档条数。 |
| `data_path` | 规则源文件根目录；增量扫描在此目录下进行。 |
| `md5_hex_store` | 已入库文件 MD5 记录文件路径，用于跳过未变更文件。 |
| `allow_knowledge_files_type` | 允许参与入库的扩展名列表，如 `.txt`、`.pdf`、`.xlsx`。 |
| `chunk_size` / `chunk_overlap` | 文本分块参数。 |
| `separators` | `RecursiveCharacterTextSplitter` 的分隔符优先级列表。 |

---

## 4. 核心方法

### 4.1 `__init__(embedding=embedding_model)`

- 读取 `chroma_conf["rule_library"]`，创建指向持久化目录的 `Chroma` 实例。
- 初始化与配置一致的 `RecursiveCharacterTextSplitter`。

### 4.2 `get_retriever()`

- 返回 `self.vec_store.as_retriever(search_kwargs={"k": self.rl["k"]})`。
- 供 `RagSummarizeService` 等对自然语言查询做 Top-K 相似片段检索。

### 4.3 `load_document()`

**流程概要：**

1. 列出 `data_path` 下所有符合 `allow_knowledge_files_type` 的文件。
2. 对每个文件计算 MD5；若 MD5 已出现在 `md5_hex_store` 文件中，则**跳过**（认为已入库）。
3. 否则按扩展名选择加载器，得到 `Document` 列表；经 `splitter.split_documents` 分块。
4. 调用 `vec_store.add_documents` 写入向量库，并将该文件 MD5 **追加**写入 `md5_hex_store`。

**异常与边界：**

- MD5 计算失败则跳过该文件。
- 加载结果为空或分块后为空会打日志并跳过。
- 单文件处理异常会记录 error 日志并 `continue`，不影响其它文件。

**注意：** MD5 列表只追加不删除；若**删除或替换**源文件后希望重新入库，需要手动维护 `md5_hex_store` 或采用团队约定的重建流程。

---

## 5. 支持的数据格式

| 扩展名 | 处理方式 |
|--------|----------|
| `.pdf` | `pdf_loader` |
| `.txt` | `text_loader` |
| `.xlsx` | `xlsx_loader` |
| 其它 | 忽略（返回空列表） |

实际规则内容可放在 `data/rule_library`（以当前 `chroma.yaml` 为准）下的表格、PDF、纯文本等，与游戏角色表、地图说明等对应。

---

## 6. 与 DecisionAgent / RagSummarizeService 的协作

- `DecisionAgent` 默认构造 `RagSummarizeService()`；`RagSummarizeService` 内部创建 `RuleLibrary()`，通过 `get_retriever()` 检索后再用 `arag_summarize` 生成规则摘要并写入决策提示词。
- 若将 `decision_agent.use_rag_summarize` 设为 `false`，`DecisionAgent` 仍通过同一 `RagSummarizeService` 暴露的 `retriever` 做原文摘录，不调用摘要链。

---

## 7. 独立运行与维护

模块末尾 `if __name__ == "__main__"` 提供最小示例：

1. 实例化 `RuleLibrary`；
2. 调用 `load_document()` 执行增量入库；
3. 获取 `retriever` 并对样例查询（如 `"超能力者"`）执行 `invoke` 并打印 `page_content`。

日常更新规则文件后，在合适环境执行一次 `load_document()` 即可将新文件同步进向量库（已记录 MD5 的未变更文件不会重复写入）。
