from time import time
from typing import Annotated, Literal
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field

class LipSyncPathRequest(BaseModel):
    image_id: str = Field(
        min_length=1,
        max_length=255,
        description="ID of the input face image",
    )
    audio_path: str = Field(
        min_length=1,
        max_length=1024,
        description="Path to the input speech audio",
    )