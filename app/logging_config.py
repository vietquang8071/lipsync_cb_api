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
    """
    Handler dùng để chuyển log từ logging chuẩn của Python sang Loguru.

    Class này giúp các thư viện đang dùng module logging, ví dụ Celery hoặc
    Kombu, ghi log qua cùng hệ thống Loguru của ứng dụng. Nhờ đó log được gom
    về cùng format, cùng file và cùng cấu hình rotation/retention.
    """

    def emit(self, record):
        """
        Ghi một bản ghi logging vào Loguru.

        Hàm này chuyển level của logging record sang level tương ứng trong
        Loguru. Nếu level không tồn tại trong Loguru, hàm dùng trực tiếp giá
        trị
        số của level. Thông tin exception, nếu có, cũng được chuyển tiếp sang
        Loguru.

        Tham số:
            record: Bản ghi log do module logging chuẩn của Python tạo ra.
        """

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def configure_loguru(log_file_name: str) -> None:
    """
    Cấu hình Loguru ghi log ra file.

    Hàm này tạo thư mục log nếu chưa tồn tại, thêm sink ghi log vào file và
    tránh
    cấu hình trùng cùng một file log nhiều lần. Các thông số như level,
    rotation,
    retention và compression có thể được cấu hình qua biến môi trường.

    Tham số:
        log_file_name: Tên file log cần ghi trong thư mục log của ứng dụng.
            Thư mục mặc định là PROJECT_DIR/logs, hoặc giá trị của biến môi
            trường LIPSYNC_LOG_DIR nếu được thiết lập.
    """

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
    """
    Cấu hình logging cho Celery worker và các thư viện liên quan.

    Hàm này cấu hình Loguru để ghi log worker ra file, sau đó gắn
    InterceptHandler vào logging chuẩn của Python. Nhờ vậy log từ Celery,
    Kombu
    và các logger nội bộ của worker được chuyển về Loguru. Hàm có cơ chế chặn
    cấu hình lặp lại nhiều lần.

    Tham số:
        log_file_name: Tên file log dành cho worker. Mặc định là "worker.log".
    """

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
    """
    Thêm middleware ghi log cho các HTTP request của FastAPI.

    Middleware này đo thời gian xử lý mỗi request, ghi log method, path,
    status_code và elapsed_ms sau khi request hoàn tất. Nếu request gây ra lỗi
    chưa được xử lý, middleware sẽ ghi exception vào log rồi raise lại lỗi để
    FastAPI tiếp tục xử lý theo cơ chế mặc định.

    Tham số:
        app: Instance FastAPI cần được gắn middleware ghi log request.
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Ghi log cho một HTTP request cụ thể.

        Hàm này được FastAPI gọi cho từng request đi qua middleware. Nó ghi
        nhận
        thời điểm bắt đầu, gọi handler tiếp theo, tính thời gian xử lý và ghi
        log
        kết quả request.

        Tham số:
            request: Đối tượng Request hiện tại của FastAPI.
            call_next: Callable dùng để chuyển request tới middleware hoặc
            route
                handler tiếp theo.

        Trả về:
            Response được trả về từ route handler hoặc middleware tiếp theo.
        """

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