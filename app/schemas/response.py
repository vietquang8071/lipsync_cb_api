from typing import Literal
from pydantic import BaseModel

class ErrorResponse(BaseModel):
  detail: str

class LipSyncVideoMeta(BaseModel):
  job_id: str
  status: Literal["completed"] = "completed"
  filename: str
  content_type: str = "video/mp4"

class LipSyncQueuedResponse(BaseModel):
    job_id: str
    task_id: str
    status: Literal["queued"] = "queued"
    video_url: str



GENERATE_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Invalid image or audio input.",
    },
    415: {
        "model": ErrorResponse,
        "description": "Unsupported media type.",
    },
    500: {
        "model": ErrorResponse,
        "description": "Failed to enqueue LipSync job.",
    },
    503: {
        "model": ErrorResponse,
        "description": "LipSync service is not initialized.",
    },
}

VIDEO_RESPONSES = {
    200: {
        "content": {
            "video/mp4": {},
        },
        "description": "Generated LipSync video.",
    },
    404: {
        "model": ErrorResponse,
        "description": "LipSync video not found or not completed.",
    },
}