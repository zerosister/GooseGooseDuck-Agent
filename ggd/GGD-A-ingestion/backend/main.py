from __future__ import annotations

import os
import subprocess
import sys
import asyncio
import logging
import queue
import signal
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Allow running this file directly: `python backend/main.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.schemas.contract import IngestionOutput, DecisionOutput
from backend.agents.ingestion import IngestionAgent

logger = logging.getLogger("ggd-a")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

B_SERVICE_URL = os.environ.get("B_SERVICE_URL", "http://127.0.0.1:9889")
_WORK_B_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "GooseGooseDuck-Agent-main", "work_B"))
_b_process: Optional[subprocess.Popen] = None


def _forward_ingestion_to_b(ingestion: IngestionOutput):
    """Fire-and-forget: send ingestion to B-side service (runs in background thread)."""
    try:
        import urllib.request
        import json
        data = json.dumps(ingestion.model_dump(), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{B_SERVICE_URL}/api/v1/ingestion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info("Forwarded ingestion to B: %s", resp.status)
    except Exception as e:
        logger.warning("Failed to forward ingestion to B: %s", e)


def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


def _start_b_service() -> Optional[subprocess.Popen]:
    """Start B-side (memory+decision) as a subprocess."""
    if not os.path.isdir(_WORK_B_DIR):
        logger.warning("work_B directory not found at %s, B-side will not auto-start", _WORK_B_DIR)
        return None
    env = {**os.environ, "PYTHONPATH": _WORK_B_DIR}
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.main"],
        cwd=_WORK_B_DIR,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    logger.info("[B-side] Started as subprocess (pid=%s, cwd=%s)", proc.pid, _WORK_B_DIR)

    import urllib.request
    for i in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"{B_SERVICE_URL}/api/v1/status", timeout=2) as resp:
                if resp.status == 200:
                    logger.info("[B-side] Service is ready")
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            logger.error("[B-side] Process exited with code %s", proc.returncode)
            return None
    logger.warning("[B-side] Timed out waiting for service to become ready")
    return proc


def _stop_b_service(proc: Optional[subprocess.Popen]):
    if proc is None or proc.poll() is not None:
        return
    logger.info("[B-side] Stopping subprocess (pid=%s) ...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    logger.info("[B-side] Subprocess stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _b_process
    asyncio.create_task(process_events())
    _b_process = await asyncio.get_event_loop().run_in_executor(None, _start_b_service)
    def _warmup_ocr():
        try:
            from legacy.extract_speaker_num import preload_ocr
            preload_ocr()
        except Exception as e:
            logger.warning("OCR pre-load failed (will retry lazily): %s", e)
    threading.Thread(target=_warmup_ocr, daemon=True).start()
    yield
    _stop_b_service(_b_process)


app = FastAPI(
    title="GGD-A Ingestion API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        threading.Thread(target=_forward_ingestion_to_b, args=(out,), daemon=True).start()

    def init(self):
        return True

    def select_window(self):
        try:
            from legacy.window_selector import select_window  # lazy import (pywin32/tk deps)
        except Exception as e:
            raise RuntimeError(
                "窗口选择不可用：缺少依赖（通常是 pywin32/tkinter）。"
                "请在 shixi 环境安装 `pywin32` 后重试。原始错误："
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
            from legacy.screen_monitor import WindowScreenMonitor  # lazy import (pywin32 deps)
            from legacy.extract_speaker_statement import GooseGooseDuckAudioAnalyzer  # lazy import (pyaudio/funasr deps)
        except Exception as e:
            raise RuntimeError(
                "启动读入监控失败：缺少依赖。请按 `GGD-A-ingestion/requirements.txt` 安装依赖。"
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
        print(f"[INIT]   - Results saved to: game_analysis.json", flush=True)
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


@app.get("/")
async def root():
    return FileResponse(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"),
        media_type="text/html",
    )


@app.get("/api/status")
async def get_status():
    return controller.status()


@app.post("/api/init")
async def init_system():
    controller.init()
    return {"status": "success", "message": "A侧服务就绪"}


@app.post("/api/debug-ocr")
async def toggle_debug_ocr(enabled: bool = True):
    """Save cropped ROI images to _debug_crops/ for verifying the OCR region."""
    from legacy.extract_speaker_num import enable_debug_save
    enable_debug_save(enabled)
    return {"status": "success", "debug_save": enabled}


@app.post("/api/select-window")
async def api_select_window():
    hwnd, title = await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
    if hwnd is None:
        return {"status": "cancelled"}
    return {"status": "success", "window": {"hwnd": hwnd, "title": title}}


@app.post("/api/start")
async def api_start(req: StartMonitoringRequest):
    if controller.hwnd is None:
        # auto prompt GUI window selector
        await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
        if controller.hwnd is None:
            return {"status": "cancelled", "message": "未选择窗口"}
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: controller.start(req.session_id))
        return {"status": "success", "window_title": controller.window_title, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def api_stop():
    await asyncio.get_event_loop().run_in_executor(None, controller.stop)
    return {"status": "success"}


class CropRegionRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int


@app.post("/api/set-crop")
async def api_set_crop(req: CropRegionRequest):
    """Set the sub-region of the window to monitor."""
    if controller._screen_monitor:
        controller._screen_monitor.set_crop_region(req.x, req.y, req.w, req.h)
        return {"status": "success", "crop": {"x": req.x, "y": req.y, "w": req.w, "h": req.h}}
    return {"status": "error", "message": "监控未启动"}


@app.post("/api/clear-crop")
async def api_clear_crop():
    if controller._screen_monitor:
        controller._screen_monitor.clear_crop_region()
    return {"status": "success"}


@app.get("/api/screenshot")
async def api_screenshot():
    """Capture current window and return as JPEG image."""
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _capture():
        from legacy.screen_monitor import ScreenCapture
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


@app.post("/api/scan-roster")
async def api_scan_roster():
    """Take screenshot and OCR extract player roster (number + name)."""
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _scan():
        from legacy.screen_monitor import ScreenCapture
        from legacy.extract_speaker_num import extract_player_roster
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


@app.get("/api/roster")
async def api_get_roster():
    return {"roster": controller.roster}


class InferenceRequest(BaseModel):
    session_id: str
    self_player_number: int = 1
    self_player_id: str = "player_1"
    role_name: str = "通灵"
    alignment: str = "goose"
    speaker_filter: Optional[list[str]] = None


@app.post("/api/inference")
async def api_inference(req: InferenceRequest):
    """Proxy decision request to B-side service and return DecisionOutput."""
    import urllib.request
    import json

    extra = {}
    if req.speaker_filter:
        extra["speaker_filter"] = req.speaker_filter

    roster_entries = [
        {"player_id": r["name"] or r["number"], "seat_number": int(r["number"]), "color": "", "metadata": {}}
        for r in controller.roster
    ] if controller.roster else []

    decision_context = {
        "session_id": req.session_id,
        "self_player_number": req.self_player_number,
        "self_player_id": req.self_player_id,
        "role_name": req.role_name,
        "alignment": req.alignment,
        "rounds": [],
        "player_roster": roster_entries,
        "extra": extra,
    }

    def _call_b():
        data = json.dumps(decision_context, ensure_ascii=False).encode("utf-8")
        r = urllib.request.Request(
            f"{B_SERVICE_URL}/api/v1/decision",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _call_b)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"B-side decision failed: {e}")

    output = DecisionOutput(
        session_id=req.session_id,
        suggestion_type="speak",
        content=result.get("speech_suggestion", ""),
        structured={
            "prior_speech_analysis": result.get("prior_speech_analysis", ""),
            "rag_queries_used": result.get("rag_queries_used", []),
            "warnings": result.get("warnings", []),
            "debug_prompt": result.get("debug_prompt", ""),
        },
        timestamp=iso_now(),
        trigger="speech",
    )
    event_queue.put({"type": "decision", "data": output.model_dump()})
    return output.model_dump()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "status", "data": controller.status().model_dump()})
        while True:
            # keepalive / ignore client messages
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9888, log_level="info")

