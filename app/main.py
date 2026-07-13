import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from app.api.v1.lipsync import router as lipsync_router
from app.model.soulX import SoulXModel
from app.service.lipsync_service import LipSyncService

APP_DIR = Path(__file__).resolve().parent
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

DEFAULT_CKPT_DIR = SOULX_DIR / "models" / "SoulX-FlashHead-1_3B"
DEFAULT_WAV2VEC_DIR = SOULX_DIR / "models" / "wav2vec2-base-960h"
DEFAULT_CONFIG_PATH = SOULX_DIR / "flash_head" / "configs" / "infer_params.yaml"
DEFAULT_STORAGE_DIR = APP_DIR.parent / "storage" / "lipsync"
DEFAULT_LOG_DIR = APP_DIR.parent / "logs"

LIPSYNC_SEED = os.getenv("SEED", 42)
MODEL_TYPE = os.getenv("MODEL_TYPE", "lite")
AUDIO_ENCODE_MODE = os.getenv("AUDIO_ENCODE_MODE", "stream")
USE_FACE_CROP = os.getenv("USE_FACE_CROP", "False")
WORLD_SIZE = os.getenv("WORLD_SIZE", 1)

def configure_logging() -> None:
  log_dir = Path(os.getenv("LIPSYNC_LOG_DIR", str(DEFAULT_LOG_DIR)))
  log_dir.mkdir(parents=True, exist_ok=True)

  logger.add(
    log_dir / "lipsync_api.log",
    level=os.getenv("LOG_LEVEL", "INFO"),
    rotation=os.getenv("LOG_ROTATION", "100 MB"),
    retention=os.getenv("LOG_RETENTION", "14 days"),
    compression=os.getenv("LOG_COMPRESSION", "zip"),
    enqueue=True,
    backtrace=True,
    diagnose=False,
  )

configure_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info("Starting LipSync API...")
  storage_dir = Path(os.getenv("LIPSYNC_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))

  app.state.lipsync_service = LipSyncService(
    storage_dir=str(storage_dir),
  )

  logger.info("LipSync API is ready")
  
  try:
    yield
  finally:
    logger.info("Shutting down LipSync API...")

app = FastAPI(
  title="LipSync API",
  version="0.1.0",
  description="GG",
  lifespan=lifespan
)

app.include_router(lipsync_router, prefix="/api/v1")

@app.get("/")
async def root():
  return {
    "service": "lipsync",
    "status": "running",
    "docs": "/docs"
  }

@app.get("/health")
async def health():
  service = getattr(
    app.state,
    "lipsync_service",
    None,
  )
  ready = (
    service is not None
    and service.storage_dir.is_dir()
  )
  return {
    "status": "ready" if ready else "not_ready",
    "component": "api",
  }