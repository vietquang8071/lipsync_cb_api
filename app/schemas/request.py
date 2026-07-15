from time import time
from typing import Annotated, Literal
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field

class LipSyncPathRequest(BaseModel):
    """
    Mô tả dữ liệu đầu vào để yêu cầu tạo video lipsync từ đường dẫn có sẵn.

    Class này được dùng làm request body cho API tạo job lipsync. Nó chứa
    thông
    tin để hệ thống xác định ảnh khuôn mặt đầu vào và file âm thanh cần dùng
    trước khi chuẩn bị job xử lý nền.

    Các trường:
        image_id: ID của ảnh khuôn mặt đầu vào. Hệ thống dùng giá trị này để
            tìm file ảnh tương ứng trong thư mục lưu trữ ảnh.
        audio_path: Đường dẫn tới file âm thanh đầu vào. File này sẽ được dùng
            làm nguồn giọng nói khi tạo video lipsync.
    """

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