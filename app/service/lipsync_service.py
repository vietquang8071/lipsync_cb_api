import os
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from loguru import logger
from starlette.concurrency import run_in_threadpool
from app.schemas.request import LipSyncPathRequest
from app.schemas.job import LipSyncJob

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
IMAGE_STORAGE = os.getenv("IMAGE_STORAGE", "images")
class LipSyncService:
  def __init__(self,
               storage_dir: str="storage/lipsync",
               ):
    self.storage_dir = Path(storage_dir)
    self.storage_dir.mkdir(parents=True, exist_ok=True)

  @staticmethod
  def _copy_upload_to_path(upload: UploadFile, output_path: Path) -> None:
      upload.file.seek(0)

      with output_path.open("wb") as buffer:
          shutil.copyfileobj(upload.file, buffer)
  
  async def prepare_job(self, request: LipSyncPathRequest) -> LipSyncJob:
    job_id = uuid4().hex
    job_dir = self.storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = self._resolve_image_path(request.image_id)
    audio_path = self._validate_audio_path(request.audio_path)
    
    output_path = job_dir/"lipsync.mp4"

    return LipSyncJob(
      job_id=job_id,
      image_path=str(image_path),
      audio_path=str(audio_path),
      output_path=str(output_path),
    )

  def _validate_audio_path(self, value: str) -> Path:
      audio_path = Path(value).resolve()
      if not audio_path.is_file():
          raise HTTPException(
              status_code=404,
              detail=f"Audio file not found: {audio_path}",
          )

      if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
          raise HTTPException(
              status_code=415,
              detail=f"Unsupported audio type: {audio_path.suffix}",
          )

      return audio_path

  def _resolve_image_path(self, image_id: str) -> Path:
      image_storage = Path(IMAGE_STORAGE)

      for suffix in IMAGE_SUFFIXES:
          candidate = image_storage / f"{image_id}{suffix}"

          if candidate.is_file():
              return candidate

      raise HTTPException(
          status_code=404,
          detail=f"Image not found for image_id: {image_id}",
      )

  # async def _save_upload(
  #     self,
  #     upload: UploadFile,
  #     target_dir:Path,
  #     filename: str,
  #     allowed_suffixes: set[str]
  # ) -> Path:
  #   suffix = Path(upload.filename or "").suffix.lower()
  #   if suffix not in allowed_suffixes:
  #     raise HTTPException(
  #       status_code=415,
  #       detail=f"Unsupported file type: {upload.filename}"
  #     ) 
  #   output_path = target_dir / f"{filename}{suffix}"
    
  #   try: 
  #     await run_in_threadpool(
  #       self._copy_upload_to_path,
  #       upload,
  #       output_path
  #     )
  #   finally:
  #     await upload.close()

  #   if output_path.stat().st_size == 0:
  #     raise HTTPException(
  #       status_code=400,
  #       detail=f"Uploaded file is empty: {upload.filename}"
  #     )
  #   return output_path