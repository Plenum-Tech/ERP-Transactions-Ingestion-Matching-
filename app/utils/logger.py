# app/logger.py
import contextvars, json, logging, os, pathlib, sys, time, uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

# ------------ request/correlation id ------------
_REQUEST_ID = contextvars.ContextVar("request_id", default="-")

def set_request_id(value: str) -> None:
    _REQUEST_ID.set(value or "-")

def request_id() -> str:
    return _REQUEST_ID.get("-")

# ------------ formatters ------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        payload: Dict[str, Any] = {
            "ts": f"{ts}.{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id()),
        }
        # include file/line for debugging
        payload["src"] = f"{record.pathname}:{record.lineno}"
        # include extra dict if provided
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = getattr(record, "request_id", request_id())
        msg = record.getMessage()
        base = f"[{record.levelname:<5}] {rid} {record.name}: {msg}"
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict) and extra:
            base += f" | {extra}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base

# ------------ core setup ------------
_CONFIGURED = False

def _build_handler(to_file: Optional[str], json_mode: bool) -> logging.Handler:
    if to_file:
        pathlib.Path(to_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(to_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.addFilter(lambda r: setattr(r, "request_id", request_id()) or True)  # inject request_id
    handler.setFormatter(JsonFormatter() if json_mode else ConsoleFormatter())
    return handler

def _level_from_env() -> int:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)

def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        json_mode = os.getenv("LOG_JSON", "false").lower() == "true"
        log_file  = os.getenv("LOG_FILE")  # e.g., logs/app.log
        root = logging.getLogger()
        root.handlers[:] = []
        root.setLevel(_level_from_env())
        root.addHandler(_build_handler(log_file, json_mode))
        _CONFIGURED = True
    return logging.getLogger(name)

# ------------ timing helper ------------
@contextmanager
def log_duration(event: str, **fields):
    log = get_logger("perf")
    start = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = round((time.perf_counter() - start) * 1000.0, 2)
        fields = dict(fields) if fields else {}
        fields["duration_ms"] = dur_ms
        log.info(event, extra={"extra": fields})

# ------------ FastAPI middleware ------------
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp
from uuid import uuid4

class RequestContextLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._log = get_logger("http")

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        set_request_id(rid)
        self._log.info("request.start", extra={"extra": {
            "method": request.method, "path": request.url.path
        }})
        with log_duration("request.duration", method=request.method, path=request.url.path):
            try:
                response = await call_next(request)
                self._log.info("request.end", extra={"extra": {
                    "status": response.status_code
                }})
                return response
            except Exception:
                self._log.exception("request.error")
                raise
            finally:
                set_request_id("-")
