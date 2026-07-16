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
    """
    Tạo chữ ký HMAC-SHA256 cho URL tải video LipSync.

    Chữ ký được tạo từ mã công việc và thời điểm hết hạn, sử dụng
    `DOWNLOAD_URL_SECRET`, nhằm ngăn client sửa đổi các tham số của URL tải xuống.

    Args:
        job_id: Mã định danh của công việc LipSync.
        expires: Thời điểm URL hết hạn, biểu diễn dưới dạng Unix timestamp.

    Returns:
        Chuỗi chữ ký HMAC-SHA256 ở định dạng hexadecimal.
    """

    message = f"lipsync-video:{job_id}:{expires}".encode()

    return hmac.new(
        DOWNLOAD_URL_SECRET,
        message,
        hashlib.sha256,
    ).hexdigest()


def create_video_download_url(job_id: str) -> tuple[str, int]:
    """
    Tạo URL có chữ ký và thời hạn để tải video LipSync.

    Args:
        job_id: Mã định danh của công việc LipSync.

    Returns:
        Tuple gồm URL tải video đã ký và thời điểm hết hạn dưới dạng
        Unix timestamp.
    """

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
  Lấy dịch vụ LipSync từ trạng thái của ứng dụng FastAPI.

  Args:
      request: Request hiện tại, dùng để truy cập `app.state`.

  Returns:
      Instance dịch vụ LipSync đã được khởi tạo.

  Raises:
      HTTPException: HTTP 503 nếu dịch vụ chưa được lưu trong
          `request.app.state`.
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
  Kiểm tra định dạng mã công việc LipSync.

  Mã hợp lệ phải dài đúng 32 ký tự và mọi ký tự phải thỏa mãn
  `str.isalnum()`.

  Args:
      job_id: Mã công việc cần kiểm tra.

  Raises:
      HTTPException: HTTP 400 nếu mã công việc không hợp lệ.
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
  Chuẩn bị và đưa một tác vụ tạo video LipSync vào hàng đợi Celery.

  Endpoint trả về ngay sau khi tác vụ được enqueue, không chờ quá trình
  tạo video hoàn tất.

  Args:
      payload: Mã ảnh và đường dẫn tệp âm thanh dùng để tạo video.
      service: Dịch vụ LipSync được cung cấp qua dependency injection.

  Returns:
      LipSyncQueuedResponse: Mã công việc, mã tác vụ Celery, trạng thái
      hàng đợi và endpoint dùng để yêu cầu URL tải xuống.

  Raises:
      HTTPException: HTTP 400 nếu mã ảnh hoặc đường dẫn âm thanh không hợp lệ.
      HTTPException: HTTP 404 nếu không tìm thấy ảnh hoặc âm thanh.
      HTTPException: HTTP 415 nếu định dạng âm thanh không được hỗ trợ.
      HTTPException: HTTP 503 nếu dịch vụ LipSync chưa được khởi tạo.
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
    """
    Cấp URL tải xuống có thời hạn cho một video LipSync đã hoàn tất.

    Hàm xác nhận tác vụ Celery đã thành công, kiểm tra tác vụ thuộc đúng
    công việc được yêu cầu và bảo đảm tệp video tồn tại trước khi tạo URL
    có chữ ký. Phản hồi được đánh dấu `Cache-Control: no-store`.

    Args:
        job_id: Mã định danh của công việc LipSync.
        task_id: Mã tác vụ Celery, nhận từ query parameter.
        response: Phản hồi FastAPI dùng để thiết lập HTTP header.
        service: Dịch vụ LipSync cung cấp thư mục lưu trữ video.

    Returns:
        LipSyncDownloadUrlResponse: URL tải video, thời điểm hết hạn và
        thời gian hiệu lực tính bằng giây.

    Raises:
        HTTPException: HTTP 400 nếu `job_id` không hợp lệ.
        HTTPException: HTTP 409 nếu tác vụ chưa hoàn tất thành công.
        HTTPException: HTTP 403 nếu tác vụ không thuộc công việc.
        HTTPException: HTTP 404 nếu tệp video không tồn tại hoặc rỗng.
        HTTPException: HTTP 503 nếu dịch vụ LipSync chưa được khởi tạo.
    """
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
  Trả về video LipSync thông qua một URL tải xuống đã ký.

  Hàm xác thực mã công việc, chữ ký HMAC và thời hạn URL trước khi kiểm
  tra và trả về tệp MP4. Phản hồi sử dụng `Cache-Control: private, no-store`.

  Args:
      job_id: Mã định danh của công việc LipSync.
      expires: Thời điểm URL hết hạn dưới dạng Unix timestamp.
      signature: Chữ ký HMAC-SHA256 hexadecimal của URL.
      service: Dịch vụ LipSync cung cấp thư mục lưu trữ video.

  Returns:
      FileResponse: Tệp video LipSync định dạng MP4.

  Raises:
      HTTPException: HTTP 400 nếu `job_id` không hợp lệ.
      HTTPException: HTTP 403 nếu chữ ký không hợp lệ.
      HTTPException: HTTP 410 nếu URL đã hết hạn.
      HTTPException: HTTP 404 nếu video không tồn tại hoặc có kích thước bằng 0.
      HTTPException: HTTP 503 nếu dịch vụ LipSync chưa được khởi tạo.
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
