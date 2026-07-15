import os 
import socket
from pathlib import Path
from time import time
from loguru import logger

from app.model.soulX import SoulXModel
from app.schemas.job import LipSyncJob
from app.worker.celery_app import celery_app
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

_model: SoulXModel | None = None

def _dump_job(job: LipSyncJob) -> dict:
  return job.model_dump() if hasattr(job, "model_dump") else job.dict()

def _get_model() -> SoulXModel:
  global _model
  if _model is not None:
    return _model
  

  _model = SoulXModel(
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
  return _model

@celery_app.task(name="process_lipsync_task", bind=True, max_retries=2)
def processing_lipsync_task(self, job_payload:dict) -> dict:
  started = time()
  job = LipSyncJob(**job_payload)
  job.status = "processing"
  job.worker_id = f"{socket.gethostname()}:{os.getpid()}"
  job.attempts = self.request.retries + 1
  job.started_at = job.started_at or time()
  job.updated_at = time()

  logger.info(
    "LipSync task started job_id={}; attempt={}; worker_id={}; image_path={}; audio_path={}; output_path={}",
    job.job_id,
    job.attempts,
    job.worker_id,
    job.image_path,
    job.audio_path,
    job.output_path,
  )

  self.update_state(state="PROGRESS", meta=_dump_job(job))

  try:

    image_path = Path(job.image_path)
    audio_path = Path(job.audio_path)

    logger.info("LipSync loading model job_id={}", job.job_id)
    model = _get_model()
    logger.info("LipSync model ready job_id={}", job.job_id)

    output_path = Path(job.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(
      "LipSync generate_video started job_id={}",
      job.job_id,
    )

    model.generate_video(
      Path(image_path),
      Path(audio_path),
      output_path=output_path,
      seed=int(os.getenv("SEED", 42)),
      use_face_crop=os.getenv("USE_FACE_CROP", default=False)
    )

    if not output_path.exists():
      raise RuntimeError("Output video was not created")
    
    logger.info(
      "LipSync generate_video completed job_id={} output_path={} output_size={} elapsed_sec={:.2f}",
      job.job_id,
      output_path,
      output_path.stat().st_size,
      time() - started,
    )


  except Exception as exc:
    will_retry = self.request.retries < self.max_retries

    logger.exception(
      "LipSync task failed job_id={} attempt={} max_retries={} will_retry={} error_type={} error={}",
      job.job_id,
      job.attempts,
      self.max_retries,
      will_retry,
      type(exc).__name__,
      str(exc),
    )

    job.error = f"{type(exc).__name__}: {exc}"
    job.updated_at = time()

    if will_retry:
      job.status = "retry"
      self.update_state(state="PROGRESS", meta=_dump_job(job))
      raise self.retry(exc=exc, countdown=10 * job.attempts)

    job.status = "failed"
    return _dump_job(job)

  job.status = "completed"
  job.completed_at = time()
  job.updated_at = job.completed_at
  return _dump_job(job)

