"""
Living AI System — FastAPI Backend
Production server: uvicorn with multiple workers.
Full async throughout. Structured JSON logging.
WebSocket streaming. Health endpoint with dependency status.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.logging_config import configure_logging
from backend.websocket.manager import WebSocketManager
from controller.main import MasterController, InputMessage

configure_logging()
log = structlog.get_logger(__name__)

controller = MasterController()
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the controller and all background systems."""
    log.info("application.starting")
    await controller.start()
    yield
    log.info("application.stopping")
    await controller.stop()


app = FastAPI(
    title="Living AI System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1, max_length=32000)
    modality: str = Field(default="text")


class ChatResponse(BaseModel):
    id: str
    session_id: str
    trace_id: str
    content: str
    modules_activated: list[str]
    confidence: float
    timestamp: str


class TaskRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    payload: dict
    priority: int = Field(default=5, ge=0, le=10)


class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    created_at: str


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


# ─── Chat Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Synchronous chat endpoint.
    Returns complete response after all modules have processed.
    Use /ws/chat for streaming responses.
    """
    trace_id = req.headers.get("X-Trace-ID", str(uuid.uuid4()))

    message = InputMessage(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=request.session_id,
        content=request.content,
        modality=request.modality,
        trace_id=trace_id,
    )

    output = await controller.process(message)

    return ChatResponse(
        id=output.id,
        session_id=output.session_id,
        trace_id=output.trace_id,
        content=output.content,
        modules_activated=output.modules_activated,
        confidence=output.confidence,
        timestamp=output.timestamp,
    )


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket streaming chat endpoint.
    Tokens stream to the client as they are generated.
    """
    await ws_manager.connect(websocket)
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_json()
            content = data.get("content", "").strip()
            if not content:
                continue

            session_id = data.get("session_id", session_id)
            trace_id = str(uuid.uuid4())

            message = InputMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=session_id,
                content=content,
                modality=data.get("modality", "text"),
                trace_id=trace_id,
            )

            # Send start signal
            await websocket.send_json({
                "type": "start",
                "trace_id": trace_id,
                "session_id": session_id,
            })

            # Stream tokens
            async for token in controller.process_stream(message):
                await websocket.send_json({
                    "type": "token",
                    "content": token,
                    "trace_id": trace_id,
                })

            # Send completion signal
            await websocket.send_json({
                "type": "done",
                "trace_id": trace_id,
                "session_id": session_id,
            })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        log.info("websocket.disconnected", session_id=session_id)
    except Exception as exc:
        log.error("websocket.error", session_id=session_id, error=str(exc))
        ws_manager.disconnect(websocket)


# ─── Task Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/tasks", response_model=TaskResponse)
async def submit_task(request: TaskRequest):
    """Submit an asynchronous task for background processing."""
    from backend.tasks.executor import TaskExecutor
    executor = TaskExecutor()
    task_id = await executor.submit(
        session_id=request.session_id,
        task_type=request.task_type,
        payload=request.payload,
        priority=request.priority,
    )
    return TaskResponse(
        task_id=task_id,
        session_id=request.session_id,
        status="queued",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Poll the status of a submitted task."""
    from backend.tasks.executor import TaskExecutor
    executor = TaskExecutor()
    status = await executor.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


# ─── Memory Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search episodic memory and knowledge base."""
    episodic = await controller.episodic_memory.retrieve(
        query=request.query,
        session_id="global",
        top_k=request.top_k,
    )
    knowledge = await controller.knowledge_base.retrieve(
        query=request.query,
        top_k=request.top_k,
    )
    return {
        "episodic_results": episodic,
        "knowledge_results": knowledge,
        "query": request.query,
    }


@app.get("/api/memory/status")
async def memory_status():
    """Return status of all memory tiers."""
    return {
        "working_memory": controller.working_memory.get_status(),
        "episodic_memory": controller.episodic_memory.get_status(),
        "knowledge_base": controller.knowledge_base.get_status(),
    }


# ─── Health Endpoint ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check with full dependency status.
    Returns 200 if system is operational.
    """
    cee_status = controller.cee.get_status() if controller.cee else {"running": False}
    memory_status = controller.working_memory.get_status()
    episodic_status = controller.episodic_memory.get_status()
    kb_status = controller.knowledge_base.get_status()
    gate_status = controller.capability_gate.get_status()

    all_healthy = (
        cee_status.get("running", False)
        and episodic_status.get("initialised", False)
        and kb_status.get("initialised", False)
    )

    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "cee": cee_status,
            "memory": {
                "working": memory_status,
                "episodic": episodic_status,
                "knowledge_base": kb_status,
            },
            "capabilities": gate_status,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        access_log=False,
    )
