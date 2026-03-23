"""Ingestion UI + monitoring API (A-side) and WebSocket."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend import app_state
from backend.agents.ingestion import IngestionAgent
from backend.schemas.contract import DecisionOutput, IngestionOutput, iso_now
from backend.schemas.decision import DecisionContext, PlayerRosterEntry

logger = logging.getLogger("ggd-a")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FRONTEND_INDEX = os.path.join(_REPO_ROOT, "frontend", "index.html")


def _forward_ingestion_to_memory(ingestion: IngestionOutput) -> None:
    """Schedule MemoryGraph.ainvoke on the main asyncio loop (called from ASR thread)."""
    g = app_state.graph
    loop = app_state.main_loop
    if g is None or loop is None:
        logger.warning("Memory graph not ready; dropping ingestion")
        return

    async def _run() -> None:
        await g.ainvoke(ingestion, ingestion.session_id)

    fut = asyncio.run_coroutine_threadsafe(_run(), loop)
    try:
        fut.result(timeout=120)
    except Exception as e:
        logger.warning("Failed to forward ingestion to memory graph: %s", e)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)


manager = ConnectionManager()
event_queue: queue.Queue[dict[str, Any]] = queue.Queue()


async def process_events():
    while True:
        try:
            while not event_queue.empty():
                event = event_queue.get_nowait()
                await manager.broadcast(event)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


router = APIRouter(tags=["ingestion"])


class StartMonitoringRequest(BaseModel):
    session_id: str = "default_session"


class StatusResponse(BaseModel):
    status: str
    is_running: bool
    current_speaker: Optional[str] = None
    window_title: Optional[str] = None
    session_id: Optional[str] = None


class MonitorController:
    def __init__(self):
        self.hwnd: Optional[int] = None
        self.window_title: Optional[str] = None
        self.is_running: bool = False
        self._lock = threading.Lock()

        self._screen_monitor = None
        self._audio_analyzer = None
        self._current_speaker: Optional[str] = None
        self._agent: Optional[IngestionAgent] = None
        self.roster: list[dict] = []

    def _on_digit_change(self, new_digit, old_digit):
        print(f"[SPEAKER] Change: {old_digit} -> {new_digit}", flush=True)
        self._current_speaker = new_digit
        event_queue.put(
            {
                "type": "speaker_change",
                "data": {"new_speaker": new_digit, "old_speaker": old_digit, "timestamp": iso_now()},
            }
        )
        if self._audio_analyzer:
            self._audio_analyzer.set_speaker(new_digit, round_num=1)

    def _on_new_record(self, record: dict[str, Any]):
        print(f"[ASR] New record: speaker={record.get('speaker')}, text={record.get('text', '')!r}", flush=True)
        if not self._agent:
            return
        speaker_id = record.get("speaker") or self._current_speaker
        emotion = record.get("emotion")
        text = record.get("text", "")
        out = IngestionOutput(
            type="speech",
            content=text,
            metadata={
                **({"speaker_id": speaker_id} if speaker_id else {}),
                **({"emotion_summary": str(emotion)} if emotion else {}),
                "source": "legacy_audio_analyzer",
            },
            timestamp=iso_now(),
            session_id=self._agent.session_id,
            sequence_id=record.get("id") or None,
        )
        event_queue.put({"type": "ingestion", "data": out.model_dump()})
        threading.Thread(target=_forward_ingestion_to_memory, args=(out,), daemon=True).start()

    def init(self):
        return True

    def select_window(self):
        try:
            from backend.legacy.window_selector import select_window  # lazy import (pywin32/tk deps)
        except Exception as e:
            raise RuntimeError(
                "窗口选择不可用：缺少依赖（通常是 pywin32/tkinter）。"
                "请安装 `pywin32` 后重试。原始错误："
                f"{e}"
            )

        hwnd, title = select_window()
        if hwnd is None:
            return None, None
        self.hwnd = int(hwnd)
        self.window_title = str(title)
        return self.hwnd, self.window_title

    def start(self, session_id: str):
        try:
            from backend.legacy.screen_monitor import WindowScreenMonitor  # lazy import (pywin32 deps)
            from backend.legacy.extract_speaker_statement import GooseGooseDuckAudioAnalyzer  # lazy import
        except Exception as e:
            raise RuntimeError(
                "启动读入监控失败：缺少依赖。请按仓库根目录 `requirements.txt` 安装依赖。"
                f" 原始错误：{e}"
            )

        with self._lock:
            if self.is_running:
                return True
            if self.hwnd is None:
                raise RuntimeError("未选择窗口")

            print(f"[INIT] Creating IngestionAgent (session={session_id}) ...", flush=True)
            self._agent = IngestionAgent(session_id=session_id)
            print(f"[INIT] Loading ASR model (FunASR SenseVoice-Small) ...", flush=True)
            self._audio_analyzer = GooseGooseDuckAudioAnalyzer(on_new_record=self._on_new_record, auto_save=True)
            print(f"[INIT] ASR model loaded OK", flush=True)
            print(f"[INIT] Creating screen monitor (hwnd={self.hwnd}, interval=0.5s) ...", flush=True)
            self._screen_monitor = WindowScreenMonitor(hwnd=self.hwnd, on_digit_change=self._on_digit_change, interval=0.5)

            self._audio_analyzer.start()
            self._screen_monitor.start()
            self.is_running = True

        print(f"[INIT] All systems started! Monitoring window: {self.window_title!r}", flush=True)
        print(f"[INIT]   - Screen capture: every 0.5s -> OCR -> detect speaker", flush=True)
        print(f"[INIT]   - Audio capture: VB-Cable -> VAD -> silence segmentation -> ASR", flush=True)
        print(f"[INIT]   - Results saved to: data/game_analysis.json", flush=True)
        event_queue.put({"type": "status_change", "data": {"status": "running", "timestamp": iso_now()}})
        return True

    def stop(self):
        with self._lock:
            if not self.is_running:
                return True
            try:
                if self._screen_monitor:
                    self._screen_monitor.stop()
                if self._audio_analyzer:
                    self._audio_analyzer.stop(round_num=1)
            finally:
                self._screen_monitor = None
                self._audio_analyzer = None
                self.is_running = False
        event_queue.put({"type": "status_change", "data": {"status": "stopped", "timestamp": iso_now()}})
        return True

    def status(self) -> StatusResponse:
        return StatusResponse(
            status="running" if self.is_running else "ready",
            is_running=self.is_running,
            current_speaker=self._current_speaker,
            window_title=self.window_title,
            session_id=self._agent.session_id if self._agent else None,
        )


controller = MonitorController()


@router.get("/")
async def root():
    return FileResponse(_FRONTEND_INDEX, media_type="text/html")


@router.get("/api/status")
async def get_status():
    return controller.status()


@router.post("/api/init")
async def init_system():
    controller.init()
    return {"status": "success", "message": "A侧服务就绪"}


@router.post("/api/debug-ocr")
async def toggle_debug_ocr(enabled: bool = True):
    """Save cropped ROI images to _debug_crops/ for verifying the OCR region."""
    from backend.legacy.extract_speaker_num import enable_debug_save

    enable_debug_save(enabled)
    return {"status": "success", "debug_save": enabled}


@router.post("/api/select-window")
async def api_select_window():
    hwnd, title = await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
    if hwnd is None:
        return {"status": "cancelled"}
    return {"status": "success", "window": {"hwnd": hwnd, "title": title}}


@router.post("/api/start")
async def api_start(req: StartMonitoringRequest):
    if controller.hwnd is None:
        await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
        if controller.hwnd is None:
            return {"status": "cancelled", "message": "未选择窗口"}
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: controller.start(req.session_id))
        return {"status": "success", "window_title": controller.window_title, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stop")
async def api_stop():
    await asyncio.get_event_loop().run_in_executor(None, controller.stop)
    return {"status": "success"}


class CropRegionRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int


@router.post("/api/set-crop")
async def api_set_crop(req: CropRegionRequest):
    """Set the sub-region of the window to monitor."""
    if controller._screen_monitor:
        controller._screen_monitor.set_crop_region(req.x, req.y, req.w, req.h)
        return {"status": "success", "crop": {"x": req.x, "y": req.y, "w": req.w, "h": req.h}}
    return {"status": "error", "message": "监控未启动"}


@router.post("/api/clear-crop")
async def api_clear_crop():
    if controller._screen_monitor:
        controller._screen_monitor.clear_crop_region()
    return {"status": "success"}


@router.get("/api/screenshot")
async def api_screenshot():
    """Capture current window and return as JPEG image."""
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _capture():
        from backend.legacy.screen_monitor import ScreenCapture
        import cv2

        cap = ScreenCapture(controller.hwnd)
        try:
            img = cap.capture()
        finally:
            cap.release()
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    jpg_bytes = await asyncio.get_event_loop().run_in_executor(None, _capture)
    if jpg_bytes is None:
        raise HTTPException(status_code=500, detail="截图失败")
    return Response(content=jpg_bytes, media_type="image/jpeg")


@router.post("/api/scan-roster")
async def api_scan_roster():
    """Take screenshot and OCR extract player roster (number + name)."""
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _scan():
        from backend.legacy.screen_monitor import ScreenCapture
        from backend.legacy.extract_speaker_num import extract_player_roster

        cap = ScreenCapture(controller.hwnd)
        try:
            img = cap.capture()
        finally:
            cap.release()
        if img is None:
            return []
        return extract_player_roster(img)

    roster = await asyncio.get_event_loop().run_in_executor(None, _scan)
    if roster:
        controller.roster = roster
        event_queue.put({"type": "roster", "data": roster})
    return {"status": "success", "roster": roster}


@router.get("/api/roster")
async def api_get_roster():
    return {"roster": controller.roster}


class InferenceRequest(BaseModel):
    session_id: str
    self_player_number: int = 1
    self_player_id: str = "player_1"
    role_name: str = "通灵"
    alignment: str = "goose"
    speaker_filter: Optional[list[str]] = None


@router.post("/api/inference")
async def api_inference(req: InferenceRequest):
    """Run decision in-process (same as POST /api/v1/decision)."""
    from backend.routers.decision import execute_decision

    extra: dict[str, Any] = {}
    if req.speaker_filter:
        extra["speaker_filter"] = req.speaker_filter

    roster_entries = [
        PlayerRosterEntry(
            player_id=r["name"] or r["number"],
            seat_number=int(r["number"]),
            color="",
            metadata={},
        )
        for r in controller.roster
    ] if controller.roster else []

    ctx = DecisionContext(
        session_id=req.session_id,
        self_player_number=req.self_player_number,
        self_player_id=req.self_player_id,
        role_name=req.role_name,
        alignment=req.alignment,
        rounds=[],
        player_roster=roster_entries,
        extra=extra,
    )

    try:
        result = await execute_decision(ctx)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Decision failed: {e}") from e

    output = DecisionOutput(
        session_id=req.session_id,
        suggestion_type="speak",
        content=result.speech_suggestion,
        structured={
            "prior_speech_analysis": result.prior_speech_analysis,
            "rag_queries_used": result.rag_queries_used,
            "warnings": result.warnings,
            "debug_prompt": result.debug_prompt or "",
        },
        timestamp=iso_now(),
        trigger="speech",
    )
    event_queue.put({"type": "decision", "data": output.model_dump()})
    return output.model_dump()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "status", "data": controller.status().model_dump()})
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)
