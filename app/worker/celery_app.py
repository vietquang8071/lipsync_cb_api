from celery import Celery
from celery.signals import worker_process_init, setup_logging
from app.logging_config import configure_loguru, configure_worker_logging
import os
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
    return os.getenv("WORKER_LOG_FILE", "worker/worker_loguru.log")


@setup_logging.connect
def setup_celery_logging(**kwargs):
    configure_worker_logging(get_worker_log_file())


@worker_process_init.connect
def configuring_worker_logging(**kwargs):
    configure_worker_logging(get_worker_log_file())
