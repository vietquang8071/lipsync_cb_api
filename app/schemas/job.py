from time import time
from typing import Annotated, Literal
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing","completed","failed","retry"]

class LipSyncJob(BaseModel):
  """
  Mô tả một job tạo video lipsync trong hệ thống.

  Class này được dùng làm payload nội bộ giữa API service và Celery worker.
  Nó lưu thông tin cần thiết để worker xử lý job, theo dõi trạng thái thực
  thi,
  số lần thử lại, lỗi phát sinh và các mốc thời gian trong vòng đời xử lý.

  Các trường:
    job_id: Mã định danh duy nhất của job, dài 32 ký tự.
    status: Trạng thái hiện tại của job, gồm queued, processing, completed,
      failed hoặc retry.
    image_path: Đường dẫn tới ảnh đầu vào dùng để tạo video lipsync.
    audio_path: Đường dẫn tới file âm thanh đầu vào.
    output_path: Đường dẫn nơi video lipsync sau khi sinh ra sẽ được lưu.
    attempts: Số lần worker đã thử xử lý job.
    max_attempts: Số lần thử xử lý tối đa được phép.
    error: Thông tin lỗi gần nhất nếu job xử lý thất bại.
    worker_id: Định danh worker đang hoặc đã xử lý job.

    created_at: Thời điểm job được tạo, tính theo Unix timestamp.
    updated_at: Thời điểm job được cập nhật gần nhất.
    started_at: Thời điểm worker bắt đầu xử lý job.
    completed_at: Thời điểm job xử lý hoàn tất thành công.
  """

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
