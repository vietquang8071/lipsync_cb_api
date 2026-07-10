from time import time
from typing import Annotated, Literal
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field

ModelType = Literal["lite"]
AudioEncodeMode = Literal["stream"]
JobStatus = Literal["queued", "processing","completed","failed","retry"]

class LipSyncOption(BaseModel):
  seed: int = Field(default=42, ge=0)
  model_type: ModelType = "lite"
  audio_encode_mode: AudioEncodeMode="stream"
  use_face_crop: bool=False

class LipSyncJob(BaseModel):
  job_id: str=Field(min_length=32, max_length=32)
  status: JobStatus = "queued"
  image_path: str
  audio_path: str
  output_path: str
  options: LipSyncOption

  attempts: int = 0
  max_attempts: int=3
  error: str | None=None
  worker_id: str | None = None

  created_at: float = Field(default_factory=time)
  updated_at: float = Field(default_factory=time)
  started_at: float | None = None
  completed_at: float | None = None


class LipSyncUploadRequest:
  def __init__(self,
               image: Annotated[UploadFile, File(description="Input face image")],
               audio: Annotated[UploadFile, File(description="Input speech audio from TTS")],
               seed: Annotated[int, Form()] = 42,
               model_type: Annotated[ModelType, Form()] = "lite",
               audio_encode_mode: Annotated[AudioEncodeMode, Form()] = "stream",
               use_face_crop: Annotated[bool, Form()] = True
               ):
    self.image = image
    self.audio = audio
    self.options = LipSyncOption(
      seed=seed,
      model_type=model_type,
      audio_encode_mode=audio_encode_mode,
      use_face_crop=use_face_crop
    )

  