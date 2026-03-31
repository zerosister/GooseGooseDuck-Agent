import uuid
import asyncio
import queue
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.analysis_graph import AnalysisEngine
from app.audio.capture_service import BackendAudioCaptureService
from app.pipeline.input_processing import InputProcessor
from utils.logger import get_logger, log_error, log_event

app = FastAPI(title="GooseGooseDuck Baseline Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("ggd-baseline")
input_processor = InputProcessor(logger=logger)
analysis_engine = AnalysisEngine(logger=logger)

session_speeches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    log_event(logger, "ws_connect", session_id, payload={"client": str(websocket.client)})
    audio_service = BackendAudioCaptureService(
        logger,
        session_id,
        sample_rate=16000,
        chunk_seconds=4.0,
    )

    def _dequeue_audio_chunk():
        try:
            return audio_service.output_queue.get(timeout=0.5)
        except queue.Empty:
            return None

    async def consume_backend_audio() -> None:
        dequeued = 0
        while True:
            chunk = await asyncio.to_thread(_dequeue_audio_chunk)
            if chunk is None:
                await asyncio.sleep(0)
                continue
            dequeued += 1
            try:
                qsz = audio_service.output_queue.qsize()
            except NotImplementedError:
                qsz = -1
            log_event(
                logger,
                "backend_audio_consumer_dequeued",
                session_id,
                payload={
                    "seq": dequeued,
                    "source": chunk.source,
                    "pcm_bytes": int(chunk.pcm16.nbytes),
                    "sample_rate": chunk.sample_rate,
                    "queue_size_approx": qsz,
                },
            )
            speech = input_processor.process_audio_pcm16(
                session_id=session_id,
                pcm16=chunk.pcm16,
                sample_rate=chunk.sample_rate,
                source=chunk.source,
            )
            text = (speech.get("text") or "").strip()
            log_event(
                logger,
                "backend_audio_consumer_asr_done",
                session_id,
                payload={
                    "source": chunk.source,
                    "text_len": len(text),
                    "appended_to_speeches": bool(text),
                },
            )
            if text:
                session_speeches[session_id].append(speech)

    audio_consumer_task: Optional[asyncio.Task] = None

    try:
        audio_service.start()
        audio_consumer_task = asyncio.create_task(consume_backend_audio())
        log_event(
            logger,
            "backend_audio_started",
            session_id,
            payload={
                "soundcard_ok": audio_service.is_available,
                "sample_rate": 16000,
                "chunk_seconds": 4.0,
            },
        )
        await websocket.send_json({"type": "session", "session_id": session_id})
        if not audio_service.is_available:
            await websocket.send_json({"type": "warning", "message": "soundcard not installed; backend audio capture unavailable"})
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            log_event(logger, "ws_receive", session_id, payload={"type": msg_type})

            if msg_type == "frame":
                frame_b64 = message.get("image_base64", "")
                if frame_b64:
                    speech = input_processor.process_frame(session_id, frame_b64)
                    if speech.get("text"):
                        session_speeches[session_id].append(speech)
            elif msg_type == "analyze":
                speeches = session_speeches.get(session_id, [])
                result = await analysis_engine.run(session_id, speeches)
                await websocket.send_json(
                    {
                        "type": "analysis_result",
                        "session_id": session_id,
                        "speech_count": len(speeches),
                        "result": result,
                    }
                )
                log_event(
                    logger,
                    "ws_send_result",
                    session_id,
                    payload={"speech_count": len(speeches), "result_keys": list(result.keys())},
                )
            elif msg_type == "reset":
                session_speeches[session_id] = []
                await websocket.send_json({"type": "reset_ok", "session_id": session_id})
                log_event(logger, "session_reset", session_id)
            else:
                await websocket.send_json({"type": "warning", "message": f"unknown type: {msg_type}"})
    except WebSocketDisconnect:
        log_event(logger, "ws_disconnect", session_id)
    except Exception as exc:
        log_error(logger, "ws_exception", session_id, str(exc))
        await websocket.close()
    finally:
        if audio_consumer_task:
            audio_consumer_task.cancel()
            try:
                await audio_consumer_task
            except asyncio.CancelledError:
                pass
        audio_service.stop()
        log_event(logger, "backend_audio_stopped", session_id)
