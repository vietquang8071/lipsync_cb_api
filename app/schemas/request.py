from time import time
from typing import Annotated, Literal
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing","completed","failed","retry"]

class LipSyncJob(BaseModel):
  job_id: str=Field(min_length=32, max_length=32)
  status: JobStatus = "queued"
  image_path: str
  audio_path: str
  output_path: str
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
               ):
    self.image = image
    self.audio = audio
  