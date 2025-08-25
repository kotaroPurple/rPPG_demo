"""Minimal FastAPI/WebSocket service (skeleton).

Expose heartbeat endpoints and a WebSocket that will stream metrics in the
future. Keep processing integration for later to avoid coupling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import json


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
        # Push single snapshot, then close (simple prototype)
        p = Path("logs/current_metrics.json")
        if p.exists():
            try:
                await ws.send_text(p.read_text())
            except Exception:
                await ws.send_json({"error": "read-failed"})
        else:
            await ws.send_json({"status": "no-metrics"})
        await ws.close()
    except WebSocketDisconnect:
        pass


@app.get("/metrics")
async def get_metrics() -> dict:
    p = Path("logs/current_metrics.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"status": "read-failed"}
    return {"status": "no-metrics"}


def main() -> None:  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
