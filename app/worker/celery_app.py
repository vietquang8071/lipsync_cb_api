from celery import Celery
import os
celery_app = Celery(
    "lipsync_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=[
        "app.worker.lipsync_task",
    ],
)