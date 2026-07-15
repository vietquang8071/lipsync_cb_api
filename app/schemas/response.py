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
    """
    Mô tả phản hồi trả về sau khi yêu cầu tạo video lipsync được đưa vào hàng
    đợi.

    Class này được dùng làm response body cho API tạo job lipsync. Nó cho
    client
    biết job đã được tiếp nhận, task xử lý nền đã được tạo, và cung cấp
    endpoint
    để lấy video sau khi quá trình xử lý hoàn tất.

    Các trường:
        job_id: Mã định danh duy nhất của job lipsync trong hệ thống.
        task_id: Mã định danh của task xử lý nền do Celery tạo ra.
        status: Trạng thái hiện tại của job tại thời điểm trả response.
            Giá trị mặc định là "queued", thể hiện job đã được đưa vào hàng
            đợi.
        video_url: Đường dẫn API dùng để truy cập video lipsync sau khi job xử
        lý xong.
    """

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