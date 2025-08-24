"""Web service placeholder.

Designed for future FastAPI/WebSocket integration to bridge the processing
core with a browser-based UI. Keeps imports out to avoid optional dependency
failures until implemented.
"""

from __future__ import annotations


def not_implemented() -> None:  # pragma: no cover - placeholder
    raise NotImplementedError("Web service is not implemented yet.")

