import os
import sys
import time
import logging
from pathlib import Path
from loguru import logger
from fastapi import Request


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_DIR / "logs"

_configured_files: set[str] = set()
_worker_logging_configured = False


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def configure_loguru(log_file_name: str) -> None:
    log_dir = Path(os.getenv("LIPSYNC_LOG_DIR", str(DEFAULT_LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    key = str(log_path)

    if key in _configured_files:
        return

    logger.add(
        log_path,
        level=os.getenv("LOG_LEVEL", "INFO"),
        rotation=os.getenv("LOG_ROTATION", "100 MB"),
        retention=os.getenv("LOG_RETENTION", "14 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    _configured_files.add(key)


def configure_worker_logging(log_file_name: str = "worker.log") -> None:
    global _worker_logging_configured

    configure_loguru(log_file_name)

    if _worker_logging_configured:
        return

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=os.getenv("LOG_LEVEL", "INFO"),
        force=True,
    )

    for logger_name in ("celery", "celery.app.trace", "celery.worker",
    "kombu"):
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        logging.getLogger(logger_name).propagate = False

    _worker_logging_configured = True


def add_request_logging_middleware(app) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled API error method={} path={}",
                request.method,
                request.url.path,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "API request method={} path={} status_code={} elapsed_ms={:.2f}",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response