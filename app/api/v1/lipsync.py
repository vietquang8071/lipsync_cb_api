import asyncio
import hmac
import os 
import time
import hashlib
from urllib.parse import urlencode
from pathlib import Path
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status, Response, Query
from fastapi.responses import FileResponse
from app.schemas.request import LipSyncPathRequest
from app.schemas.response import GENERATE_ERROR_RESPONSES, VIDEO_RESPONSES, LipSyncQueuedResponse, LipSyncDownloadUrlResponse
from app.worker.lipsync_task import processing_lipsync_task
from app.worker.celery_app import celery_app

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")
DOWNLOAD_URL_SECRET = os.environ["DOWNLOAD_URL_SECRET"].encode()
DOWNLOAD_URL_TTL_SECONDS = int(
    os.getenv("DOWNLOAD_URL_TTL_SECONDS", "300")
)

router = APIRouter(prefix="/lipsync", tags=["lipsync"])


def create_video_signature(job_id: str, expires: int) -> str:
    message = f"lipsync-video:{job_id}:{expires}".encode()

    return hmac.new(
        DOWNLOAD_URL_SECRET,
        message,
        hashlib.sha256,
    ).hexdigest()


def create_video_download_url(job_id: str) -> tuple[str, int]:
    expires = int(time.time()) + DOWNLOAD_URL_TTL_SECONDS
    signature = create_video_signature(job_id, expires)

    query = urlencode({
        "expires": expires,
        "signature": signature,
    })

    url = (
        f"{PUBLIC_BASE_URL}"
        f"/api/v1/lipsync/jobs/{job_id}/video"
        f"?{query}"
    )

    return url, expires



def get_lipsync_service(request: Request):
  """
  Lấy instance `LipSyncService` đã được khởi tạo và lưu trong trạng thái của ứng dụng FastAPI. Hàm này thường được sử dụng làm dependency cho các endpoint. Service được lấy từ `request.app.state.lipsync_service`, vốn cần được thiết lập trước đó trong quá trình khởi động ứng dụng.
  Args:
    request: Đối tượng request hiện tại của FastAPI, được dùng để truy cập trạng thái dùng chung của ứng dụng.

  Returns:
    Instance `LipSyncService` đã được khởi tạo.

  Raises:
    HTTPException: Trả về HTTP 503 nếu `LipSyncService` chưa được khởi tạo hoặc chưa được lưu trong `app.state`.

  """
  service = getattr(request.app.state, "lipsync_service", None)
  if service is None:
    raise HTTPException(
      status_code=503,
      detail="LipSync service is not initialized.",
    )
  return service

def validate_job_id(job_id:str) -> None:
  """
  Kiểm tra tính hợp lệ của mã định danh công việc.

  Một `job_id` hợp lệ phải có đúng 32 ký tự và chỉ bao gồm
  các ký tự chữ cái hoặc chữ số.

  Args:
      job_id: Mã định danh của công việc cần kiểm tra.

  Raises:
      HTTPException: Nếu `job_id` không có đúng 32 ký tự hoặc
          chứa ký tự không phải chữ cái hay chữ số.
  """
  if len(job_id) != 32 or not job_id.isalnum():
    raise HTTPException(
      status_code=400,
      detail="Invalid job id"
    )


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED, response_model=LipSyncQueuedResponse, responses=GENERATE_ERROR_RESPONSES)
async def generate_lipsync_video(
  payload: LipSyncPathRequest,
  service=Depends(get_lipsync_service),
):
  """
  Tạo và đưa một tác vụ sinh video LipSync vào hàng đợi xử lý.

  Hàm chuẩn bị thông tin công việc từ dữ liệu đầu vào, chuyển công việc
  sang dạng dictionary để có thể tuần tự hóa, sau đó gửi tác vụ đến
  Celery worker để xử lý bất đồng bộ.

  Endpoint trả về ngay sau khi tác vụ được đưa vào hàng đợi mà không chờ
  quá trình sinh video hoàn tất.

  Args:
      payload: Dữ liệu đầu vào chứa đường dẫn ảnh, âm thanh và các tùy chọn
          cần thiết để tạo video LipSync.
      service: Dịch vụ LipSync được FastAPI cung cấp thông qua dependency
          injection, dùng để chuẩn bị và quản lý công việc.

  Returns:
      LipSyncQueuedResponse: Thông tin tác vụ đã được đưa vào hàng đợi,
      bao gồm mã công việc, mã tác vụ Celery, trạng thái hiện tại và URL
      dùng để truy cập video sau khi xử lý hoàn tất.

  Raises:
      HTTPException: Nếu dữ liệu đầu vào không hợp lệ hoặc dịch vụ LipSync
          không thể chuẩn bị công việc.
  """
  job = await service.prepare_job(payload)
  job_payload = (
    job.model_dump()
    if hasattr(job, "model_dump")
    else job.dict()
  )
  task = processing_lipsync_task.delay(job_payload)

  return LipSyncQueuedResponse(
      job_id=job.job_id,
      task_id=task.id,
      status="queued",
      download_url_endpoint=(
          f"{PUBLIC_BASE_URL}"
          f"/api/v1/lipsync/jobs/{job.job_id}/download-url"
      ),
  )

@router.post(
    "/jobs/{job_id}/download-url",
    response_model=LipSyncDownloadUrlResponse,
)
async def create_lipsync_download_url(
    job_id: str,
    task_id:str,
    response: Response,
    service=Depends(get_lipsync_service),
):
    validate_job_id(job_id)

    task = AsyncResult(
        task_id,
        app=celery_app,
    )

    if task.state != "SUCCESS":
        raise HTTPException(
            status_code=409,
            detail="LipSync task is not completed",
        )

    task_info = task.info

    if (
        not isinstance(task_info, dict)
        or task_info.get("job_id") != job_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Task does not belong to this job",
        )

    video_path = (
        Path(service.storage_dir)
        / job_id
        / "lipsync.mp4"
    )

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise HTTPException(
            status_code=404,
            detail="LipSync video not found",
        )

    download_url, expires = create_video_download_url(job_id)

    response.headers["Cache-Control"] = "no-store"

    return LipSyncDownloadUrlResponse(
        download_url=download_url,
        expires_at=expires,
        expires_in=DOWNLOAD_URL_TTL_SECONDS,
    )



@router.get("/tasks/{task_id}")
async def get_lipsync_task_status(task_id:str):
  """
  Lấy trạng thái hiện tại của một tác vụ LipSync trong Celery.

  Hàm sử dụng `task_id` để truy vấn thông tin tác vụ từ Celery backend.
  Kết quả trả về luôn bao gồm mã tác vụ và trạng thái hiện tại. Nếu tác vụ
  có thông tin bổ sung, chẳng hạn tiến trình xử lý, kết quả hoặc chi tiết
  lỗi, dữ liệu này sẽ được trả về trong trường `meta`.

  Args:
      task_id: Mã định danh của tác vụ Celery cần kiểm tra.

  Returns:
      dict: Thông tin trạng thái của tác vụ, bao gồm:

      - `task_id`: Mã định danh của tác vụ.
      - `state`: Trạng thái hiện tại của tác vụ
      - `meta`: Thông tin bổ sung của tác vụ nếu có.
  """
  result = AsyncResult(task_id, app=celery_app)
  response = {
    "task_id":task_id,
    "state": result.state,
  }
  info = result.info
  if isinstance(info, BaseException):
    response["meta"] = {
        "status": (
            "failed"
            if result.state == "FAILURE"
            else "retry"
        ),
        "error_type": type(info).__name__,
        "error": str(info)
    }
  elif info is not None:
    response["meta"] = info
  return response


@router.get("/jobs/{job_id}/video",response_class=FileResponse, responses=VIDEO_RESPONSES)
async def get_lipsync_video(
  job_id: str,
  expires: int = Query(..., gt=0),
  signature: str = Query(..., min_length=64, max_length=64),
  service=Depends(get_lipsync_service),
):
  """
  Trả về tệp video LipSync đã được tạo cho một công việc cụ thể.

  Hàm kiểm tra tính hợp lệ của `job_id`, xác định đường dẫn đến video
  trong thư mục lưu trữ của dịch vụ và kiểm tra xem tệp video đã tồn tại
  cũng như có dữ liệu hay chưa. Nếu video chưa được tạo xong hoặc không
  tồn tại, endpoint trả về lỗi HTTP 404.

  Args:
      job_id: Mã định danh của công việc LipSync cần lấy video.
      service: Dịch vụ LipSync được FastAPI cung cấp thông qua dependency
          injection, dùng để truy cập thư mục lưu trữ dữ liệu.

  Returns:
      FileResponse: Tệp video LipSync định dạng MP4 được trả về cho client.

  Raises:
      HTTPException: Nếu `job_id` không hợp lệ, video không tồn tại,
          chưa được tạo hoàn tất hoặc tệp video có kích thước bằng 0.
  """
  validate_job_id(job_id)

  expected_signature = create_video_signature(
     job_id=job_id,
     expires=expires
  )

  if not hmac.compare_digest(signature, expected_signature):
      raise HTTPException(
          status_code=403,
          detail="Invalid download signature",
      )

  if int(time.time()) > expires:
      raise HTTPException(
          status_code=410,
          detail="Download URL has expired",
      )

  video_path = Path(service.storage_dir) / job_id / f"lipsync.mp4"

  if (
      not video_path.is_file()
      or video_path.stat().st_size == 0
  ):
      raise HTTPException(
          status_code=404,
          detail="LipSync video not found or not completed",
      )

  return FileResponse(
      path=str(video_path),
      media_type="video/mp4",
      filename=f"lipsync-{job_id}.mp4",
      headers={"Cache-Control":"private, no-store"}
  )
