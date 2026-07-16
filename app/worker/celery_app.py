from dotenv import load_dotenv

load_dotenv()

from celery import Celery
from celery.signals import worker_process_init, setup_logging
from app.logging_config import configure_loguru, configure_worker_logging
import os
"""
Cấu hình Celery app cho worker xử lý job lipsync.

File này khởi tạo Celery application dùng cho hệ thống lipsync, bao gồm cấu hình
broker, result backend, danh sách module task cần load và logging cho worker.
Broker và backend mặc định dùng Redis, nhưng có thể được thay đổi thông qua biến
môi trường CELERY_BROKER_URL và CELERY_RESULT_BACKEND.

Ngoài ra, file còn đăng ký các signal của Celery để cấu hình Loguru cho worker
khi Celery thiết lập logging và khi worker process được khởi tạo.
"""

celery_app = Celery(
    "lipsync_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/10"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/11"),
    include=[
        "app.worker.lipsync_task",
    ],
)

celery_app.conf.update(
    worker_redirect_stdouts=True,
    worker_redirect_stdouts_level="INFO",
)

def get_worker_log_file() -> str:
    """
    Lấy tên file log dành cho Celery worker.

    Giá trị được đọc từ biến môi trường WORKER_LOG_FILE. Nếu biến này không được
    thiết lập, hệ thống dùng file mặc định "worker/worker_loguru.log".

    Trả về:
        Tên file log mà worker sẽ ghi log vào.
    """
    return os.getenv("WORKER_LOG_FILE", "worker/worker_loguru.log")


@setup_logging.connect
def setup_celery_logging(**kwargs):
    """
    Cấu hình logging cho Celery khi Celery thiết lập hệ thống logging.

    Hàm này được gọi thông qua signal setup_logging của Celery. Nó chuyển logging
    của worker sang cấu hình Loguru dùng chung trong ứng dụng.

    Tham số:
        **kwargs: Các tham số do Celery truyền vào khi phát signal setup_logging.
    """
    configure_worker_logging(get_worker_log_file())


@worker_process_init.connect
def configuring_worker_logging(**kwargs):
    """
    Cấu hình lại logging khi một worker process được khởi tạo.

    Hàm này đảm bảo mỗi process của Celery worker đều dùng cấu hình Loguru đúng,
    đặc biệt trong môi trường worker chạy nhiều process.

    Tham số:
        **kwargs: Các tham số do Celery truyền vào khi phát signal
            worker_process_init.
    """
    configure_worker_logging(get_worker_log_file())
