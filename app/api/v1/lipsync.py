from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from app.schemas.request import LipSyncUploadRequest
from app.schemas.response import RESPONSE_STATUS
from app.worker.lipsync_task import processing_lipsync_task
from app.worker.celery_app import celery_app

router = APIRouter(prefix="/lipsync", tags=["lipsync"])

def get_lipsync_service(request: Request):
  service = getattr(request.app.state, "lipsync_service", None)
  if service is None:
    raise HTTPException(
      status_code=503,
      detail="LipSync service is not initialized.",
    )
  return service

def validate_job_id(job_id:str) -> None:
  if len(job_id) != 32 or not job_id.isalnum():
    raise HTTPException(
      status_code=400,
      detail="Invalid job id"
    )
  
  

@router.post("/generate", responses=RESPONSE_STATUS)
async def generate_lipsync_video(
  payload: LipSyncUploadRequest=Depends(),
  service=Depends(get_lipsync_service),
):
  job = await service.prepare_job(payload)
  task = processing_lipsync_task.delay(job.dict())

  return {
    "job_id":job.job_id,
    "task_id":task.id,
    "status":"queued",
    "video_url":f"/api/v1/lipsync/jobs/{job.job_id}/video"

  }

@router.get("/tasks/{task_id}")
async def get_lipsync_task_status(task_id:str):
  result = AsyncResult(task_id, app=celery_app)
  response = {
    "task_id":task_id,
    "state": result.state,
  }
  if result.info:
    response["meta"] = result.info
  return response


@router.get("/jobs/{job_id}/video")
async def get_lipsync_video(
  job_id: str,
  service=Depends(get_lipsync_service),
):
  validate_job_id(job_id)

  video_path = Path(service.storage_dir) / job_id / "lipsync.mp4"

  if not video_path.exists():
    raise HTTPException(
      status_code=404,
      detail="LipSync video not found"
    )
  
  return FileResponse(
    path=str(video_path),
    media_type="video/mp4",
    filename="lipsync.mp4"
  )