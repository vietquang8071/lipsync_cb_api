from typing import Literal
from pydantic import BaseModel

class ErrorResponse(BaseModel):
  detail: str

class LipSyncVideoMeta(BaseModel):
  job_id: str
  status: Literal["completed"] = "completed"
  filename: str
  content_type: str = "video/mp4"

RESPONSE_STATUS = {
  200: {
    "content": {
      "video/mp4":{}
    },
    "description": "Generated LipSync video."
  }, 
  400: {
    "model": ErrorResponse,
    "description": "Invalid image/audio_input"
  },
  500: {
    "model":ErrorResponse,
    "description": "LipSync generation failed"
  },
  415: {"model": ErrorResponse, "description": "Unsupported media type."},
  503: {"model": ErrorResponse, "description": "LipSync service is not initialized."},
}