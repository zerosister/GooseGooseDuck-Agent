"""
联合测试：meeting_content → MemoryGraph（仅 memory）；「决策信号」单独传入 DecisionContext。

在 work_B 目录下：
  set PYTHONPATH=.
  python test/run_memory_decision_graph.py
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_WORK_B = Path(__file__).resolve().parent.parent
if str(_WORK_B) not in sys.path:
    sys.path.insert(0, str(_WORK_B))

from backend.agents.my_graph import MemoryGraph
from backend.schemas.contract import IngestionOutput
from backend.schemas.decision import DecisionContext, PlayerRosterEntry

_LINE_NEW = re.compile(
    r"^(\d+)号(?:（([^）]*)）)?\s*[:：]\s*(.*)$",
    re.DOTALL,
)


def _speaker_id_for_seat(
    roster: list[PlayerRosterEntry] | None, seat: int
) -> str:
    if roster:
        for p in roster:
            if p.seat_number == seat:
                return p.player_id
    return str(seat)


def load_ingestions_from_meeting_file(
    session_id: str,
    path: Path | None = None,
    base_time: datetime | None = None,
    roster: list[PlayerRosterEntry] | None = None,
) -> list[IngestionOutput]:
    file_path = path or (_WORK_B / "test" / "meeting_content.txt")
    text = file_path.read_text(encoding="utf-8")
    base = base_time or datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
    out: list[IngestionOutput] = []
    seq = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        speaker_id: str
        content: str
        meta_extra: dict[str, str | int] = {}

        m = _LINE_NEW.match(line)
        if m:
            seat = int(m.group(1))
            tag = (m.group(2) or "").strip()
            content = (m.group(3) or "").strip()
            speaker_id = _speaker_id_for_seat(roster, seat)
            meta_extra["seat_number"] = seat
            if tag:
                meta_extra["voice_tag"] = tag
        elif "|" in line:
            speaker_id, content = line.split("|", 1)
            speaker_id = speaker_id.strip()
            content = content.strip()
        else:
            raise ValueError(f"无法解析行: {raw!r}")

        if not content:
            raise ValueError(f"无效行（正文为空）: {raw!r}")

        seq += 1
        ts = (base + timedelta(seconds=seq)).isoformat()
        metadata = {"speaker_id": speaker_id, **meta_extra}
        out.append(
            IngestionOutput(
                type="speech",
                content=content,
                metadata=metadata,
                timestamp=ts,
                session_id=session_id,
                sequence_id=seq,
            )
        )
    return out


def build_player_roster() -> list[PlayerRosterEntry]:
    colors = ["白", "蓝", "绿", "粉", "红", "黄", "橙", "棕", "黑", "紫", "浅绿", "浅蓝", "浅紫"]
    ids = [
        "桌子不棋邓紫",
        "菠萝披萨W",
        "哈吉贵",
        "木槑",
        "初逢你",
        "缚清不是番茄",
        "俏坡",
        "我是少骡",
        "婴儿蓝",
        "甄果粒.",
        "辞牙乐",
        "呆呆落",
        "快叫泰哥",
    ]
    roster = []
    for i in range(1, len(colors) + 1):
        roster.append(
            PlayerRosterEntry(
                player_id=f"{ids[i - 1]}",
                seat_number=i,
                color=colors[i - 1],
            )
        )
    return roster


def build_test_decision_context(
    session_id: str,
    ingestions: list[IngestionOutput],
    self_player_number: int = 3,
    self_player_id: str = "哈吉贵",
    role_name: str = "通灵",
    alignment: str = "goose",
) -> DecisionContext:
    """与图无关：可早在会话开始由业务侧构造；决策信号到达时再与 checkpoint 组合。"""
    roster = build_player_roster()
    self_id = self_player_id or (roster[0].player_id if roster else "1")
    return DecisionContext(
        session_id=session_id,
        self_player_number=self_player_number,
        self_player_id=self_id,
        role_name=role_name,
        alignment=alignment,
        rounds=[],
        player_roster=roster,
    )


async def run_test() -> None:
    from backend.services.meeting_memory_service import ShortTermMemoryStore

    session_id = f"test_{uuid.uuid4().hex[:8]}"
    store = ShortTermMemoryStore()
    async with store.get_saver() as saver:
        app = MemoryGraph(saver)
        config = {"configurable": {"thread_id": session_id}}

        roster = build_player_roster()
        test_inputs = load_ingestions_from_meeting_file(
            session_id, roster=roster
        )
        expected_n = len(test_inputs)
        if expected_n == 0:
            print("⚠️ meeting_content.txt 无有效发言行，跳过。")
            return

        print(f"载入 {expected_n} 条发言（每条仅 memory_agent，checkpoint 不存 DecisionContext）…")
        for i, ingestion in enumerate(test_inputs):
            await app.ainvoke(ingestion, session_id)
            print(
                f"  第 {i + 1}/{expected_n} 条（speaker={ingestion.metadata.get('speaker_id')}）"
            )

        decision_ctx = build_test_decision_context(session_id, test_inputs)
        print("\n>>> 独立决策信号：run_decision(thread_id, decision_context) …")
        decision_out = await app.run_decision(session_id, decision_ctx)

        final_state = await app.graph.aget_state(config)
        values = final_state.values
        accumulated = values.get("ingestions", [])
        length = len(accumulated)
        print("\n--- 记忆累积 ---")
        print(f"预期 ingestions 长度: {expected_n}，实际: {length}")

        summary = values.get("summary")
        if summary and getattr(summary, "player_summaries", None):
            print("\n--- MemoryAgent 摘要 ---")
            for j, ps in enumerate(summary.player_summaries):
                print(f"  [{j}] speaker={ps.speaker_id}: {ps.latest_stance[:120]}...")

        dr_from_checkpoint = values.get("decision_result")
        print("\n--- DecisionAgent 输出 ---")
        out = decision_out if decision_out is not None else dr_from_checkpoint
        if out:
            print("【前序发言分析】\n", out.prior_speech_analysis[:500])
            print("\n【发言建议】\n", out.speech_suggestion[:500])
            if out.warnings:
                print("\nwarnings:", out.warnings)
        else:
            print("（无决策结果：检查 summary / DASHSCOPE / RAG）")


if __name__ == "__main__":
    start_time = time.time()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_test())
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")
