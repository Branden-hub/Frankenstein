"""
Living AI System — WebSocket Connection Manager
Manages all active WebSocket connections.
Handles connect, disconnect, and broadcast.
"""

import structlog
from fastapi import WebSocket

log = structlog.get_logger(__name__)


class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info("websocket.connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info("websocket.disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)
