## GGD-A-ingestion（分工A）

该目录是从 `GGD-AI-main` 中**分离**出来的 **A：读入 Agent** 功能集合（语音/画面/发言者匹配/情绪），用于与 B（记忆+决策）按契约对接。

### A 的输出契约

- **A → B**：`backend/schemas/contract.py` 中的 `IngestionOutput`

### 代码来源与复用

- `legacy/`：从 `GGD-AI-main` 复制的可复用实现（不改动原工程，避免破坏完整可运行版本）
- `backend/`：按 agent-plan 重新组织后的 A 侧模块（对外输出 `IngestionOutput`）

### 目录结构

```
GGD-A-ingestion/
├── backend/
│   ├── agents/ingestion.py
│   ├── schemas/contract.py
│   └── services/
├── legacy/
│   ├── extract_speaker_num.py
│   ├── extract_speaker_statement.py
│   └── screen_monitor.py
└── requirements.txt
```

