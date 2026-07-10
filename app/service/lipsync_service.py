import asyncio
import shutil
import subprocess
from collections import deque
from pathlib import Path
from uuid import uuid4
import imageio
import librosa
import numpy as np
from fastapi import HTTPException, UploadFile
from loguru import logger
from starlette.concurrency import run_in_threadpool
from app.schemas.request import LipSyncUploadRequest, LipSyncJob
from app.model.soulX import SoulXModel
from app.utils import save_video
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}

class LipSyncService:
  def __init__(self,
               model: SoulXModel,
               storage_dir: str="storage/lipsync",
               use_face_crop: bool = True
               ):
    self.model = model
    self.storage_dir = Path(storage_dir)
    self.storage_dir.mkdir(parents=True, exist_ok=True)
    self.use_face_crop=use_face_crop

  @staticmethod
  def _copy_upload_to_path(upload: UploadFile, output_path: Path) -> None:
      upload.file.seek(0)

      with output_path.open("wb") as buffer:
          shutil.copyfileobj(upload.file, buffer)
  
  async def prepare_job(self, request: LipSyncUploadRequest) -> LipSyncJob:
    if request.options.model_type != self.model.model_type:
      raise HTTPException(
        status_code=400,
        detail=f"Loaded model is {self.model.model_type} while request uses {request.options.model_type}.",
      )
    
    if request.options.audio_encode_mode != "stream":
      raise HTTPException(
        status_code=400,
        detail="Only stream audio_encode_mode is supported by this service"
      )
    job_id = uuid4().hex
    job_dir = self.storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = await self._save_upload(
      upload=request.image,
      target_dir=job_dir,
      filename="file_image",
      allowed_suffixes=IMAGE_SUFFIXES
    )

    audio_path = await self._save_upload(
      upload=request.audio,
      target_dir=job_dir,
      filename="input_audio",
      allowed_suffixes=AUDIO_SUFFIXES
    )

    output_path = job_dir/"lipsync.mp4"

    return LipSyncJob(
      job_id=job_id,
      image_path=str(image_path),
      audio_path=str(audio_path),
      output_path=str(output_path),
      options=request.options
    )
  
  async def generate_job(self, job:LipSyncJob) -> Path:
    output_path = Path(job.output_path)
    image_path = Path(job.image_path)
    audio_path = Path(job.audio_path)

    try:
      await run_in_threadpool(
        self.model.generate_video,
        image_path,
        audio_path,
        output_path,
        job.options.seed,
        job.options.use_face_crop,
      )
    except HTTPException:
      raise
    except Exception as exc:
      logger.exception("LipSync generation failed")
      raise HTTPException(
        status_code=500,
        detail=f"LipSync generation failed: {exc}"
      ) from exc
    
    if not output_path.exists():
      raise HTTPException(
        status_code=500,
        detail="output video was not created"
      )
    return output_path

  async def generate(self, request: LipSyncUploadRequest) -> Path:
    job = await self.prepare_job(request)
    return await self.generate_job(job)
  
  async def _save_upload(
      self,
      upload: UploadFile,
      target_dir:Path,
      filename: str,
      allowed_suffixes: set[str]
  ) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
      raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type: {upload.filename}"
      ) 
    output_path = target_dir / f"{filename}{suffix}"
    
    try: 
      await run_in_threadpool(
        self._copy_upload_to_path,
        upload,
        output_path
      )
    finally:
      await upload.close()

    if output_path.stat().st_size == 0:
      raise HTTPException(
        status_code=400,
        detail=f"Uploaded file is empty: {upload.filename}"
      )
    return output_path
