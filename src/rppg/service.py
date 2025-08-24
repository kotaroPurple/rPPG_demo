"""Minimal FastAPI/WebSocket service (skeleton).

Expose heartbeat endpoints and a WebSocket that will stream metrics in the
future. Keep processing integration for later to avoid coupling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:  # pragma: no cover - skeleton
    # Initialize shared state/resources here if needed
    yield
    # Cleanup resources here


app = FastAPI(title="rPPG Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:  # pragma: no cover - trivial
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_metrics(ws: WebSocket) -> None:  # pragma: no cover - skeleton
    await ws.accept()
    try:
        await ws.send_json({"hello": "rPPG"})
        # In future: push metrics periodically
        await ws.close()
    except WebSocketDisconnect:
        pass


def main() -> None:  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
