import os 
import socket
from pathlib import Path
from time import time
from loguru import logger

from app.model.soulX import SoulXModel
from app.schemas.request import LipSyncJob
from app.worker.celery_app import celery_app
from app.service.lipsync_service import LipSyncService

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

_service: LipSyncService | None = None

def _dump_job(job: LipSyncJob) -> dict:
  return job.model_dump() if hasattr(job, "model_dump") else job.dict()

def _get_service() -> LipSyncService:
  global _service
  if _service is not None:
    return _service
  

  model = SoulXModel(
    cpkt_dir=os.getenv(
        "SOULX_CKPT_DIR",
        str(SOULX_DIR / "models" / "SoulX-FlashHead-1_3B"),
    ),
    wav2vec_dir=os.getenv(
        "SOULX_WAV2VEC_DIR",
        str(SOULX_DIR / "models" / "wav2vec2-base-960h"),
    ),
    model_type=os.getenv("SOULX_MODEL_TYPE", "lite"),
    config_path=os.getenv(
        "SOULX_CONFIG_PATH",
        str(SOULX_DIR / "flash_head" / "configs" / "infer_params.yaml"),
    ),
    world_size=int(os.getenv("WORLD_SIZE", "1")),
  )

  _service = LipSyncService(
    model=model,
    storage_dir=os.getenv(
      "LIPSYNC_STORAGE_DIR",
      str(PROJECT_DIR / "storage" / "lipsync"),
    ),
    use_face_crop=False
  )

  return _service

@celery_app.task(name="process_lipsync_task", bind=True, max_retries=2)
def processing_lipsync_task(self, job_payload:dict) -> dict:
  job = LipSyncJob(**job_payload)
  job.status = "processing"
  job.worker_id = f"{socket.gethostname()}:{os.getpid()}"
  job.attempts = self.request.retries + 1
  job.started_at = job.started_at or time()
  job.updated_at = time()

  self.update_state(state="PROGRESS", meta=_dump_job(job))

  try:
    service = _get_service()
    output_path = Path(job.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    service.model.generate_video(
      Path(job.image_path),
      Path(job.audio_path),
      output_path=output_path,
      seed=job.options.seed,
      use_face_crop=job.options.use_face_crop
    )

    if not output_path.exists():
      raise RuntimeError("Output video was not created")
    
  except Exception as exc:
    logger.exception("LipSync Celery task failed")
    job.error = str(exc)
    job.updated_at = time()

    if self.request.retries < self.max_retries:
      job.status = "retry"
      self.update_state(state="PROGRESS", meta=_dump_job(job))
      raise self.retry(exc=exc, countdown=10 * job.attempts)
    job.status="failed"
    return _dump_job(job)
  
  job.status = "completed"
  job.completed_at=time()
  job.updated_at=job.completed_at
  return _dump_job(job)

