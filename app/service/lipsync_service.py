import os
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from loguru import logger
from starlette.concurrency import run_in_threadpool
from app.schemas.request import LipSyncPathRequest
from app.schemas.job import LipSyncJob

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
IMAGE_STORAGE = os.getenv("IMAGE_STORAGE", "images")
class LipSyncService:

    """
    Quản lý bước chuẩn bị job tạo video lipsync.

    LipSyncService không trực tiếp sinh video, mà chuẩn hóa và kiểm tra các đầu
    vào
    cần thiết trước khi tạo LipSyncJob. Service sẽ tạo job_id, tạo thư mục lưu
    trữ
    cho job, resolve đường dẫn ảnh từ image_id, validate đường dẫn file audio và
    thiết lập đường dẫn file video kết quả.

    Các trường:
        storage_dir: Đường dẫn thư mục gốc để lưu dữ liệu của các job lipsync.
        Giá trị mặc định là "storage/lipsync". Khi khởi tạo service, thư mục này
        sẽ được tạo nếu chưa tồn tại.
    """

    def __init__(self,
               storage_dir: str="storage/lipsync",
               ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy_upload_to_path(upload: UploadFile, output_path: Path) -> None:
        """
        Sao chép nội dung file upload vào một đường dẫn trên hệ thống file.

        Hàm này đưa con trỏ đọc của file upload về đầu file, sau đó ghi toàn bộ
        nội dung upload vào file đích. Đây là hàm đồng bộ, thường được gọi qua
        threadpool nếu dùng trong luồng async.

        Tham số:
          upload: File được upload từ client.
          output_path: Đường dẫn file đích nơi nội dung upload sẽ được lưu.
        """

        upload.file.seek(0)

        with output_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
  
    async def prepare_job(self, request: LipSyncPathRequest) -> LipSyncJob:
        """
        Chuẩn bị một job lipsync từ request đầu vào.

        Hàm này tạo job_id mới, tạo thư mục riêng cho job, tìm đường dẫn ảnh đầu
        vào
        dựa trên image_id, kiểm tra tính hợp lệ của file âm thanh, xác định đường
        dẫn
        video đầu ra và trả về đối tượng LipSyncJob để đưa vào hàng đợi xử lý nền.

        Tham số:
            request: Dữ liệu request tạo lipsync, bao gồm image_id của ảnh đầu vào
            và
            audio_path của file âm thanh.

        Trả về:
            LipSyncJob chứa thông tin đầy đủ để worker xử lý job lipsync.
        """

        job_id = uuid4().hex
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = self._resolve_image_path(request.image_id)
        audio_path = self._validate_audio_path(request.audio_path)
        
        output_path = job_dir/"lipsync.mp4"

        return LipSyncJob(
        job_id=job_id,
        image_path=str(image_path),
        audio_path=str(audio_path),
        output_path=str(output_path),
        )

    def _validate_audio_path(self, value: str) -> Path:
        """
        Kiểm tra và chuẩn hóa đường dẫn file âm thanh đầu vào.

        Hàm này chuyển đường dẫn audio sang dạng tuyệt đối, kiểm tra file có tồn
        tại hay không và kiểm tra phần mở rộng có thuộc nhóm định dạng được hỗ
        trợ.
        Nếu file không tồn tại hoặc không đúng định dạng, hàm sẽ raise
        HTTPException.

        Tham số:
          value: Đường dẫn tới file âm thanh đầu vào.

        Trả về:
          Path tuyệt đối tới file âm thanh hợp lệ.
        """


        audio_path = Path(value).resolve()
        if not audio_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found: {audio_path}",
            )

        if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported audio type: {audio_path.suffix}",
            )

        return audio_path

    def _resolve_image_path(self, image_id: str) -> Path:

        """
        Tìm đường dẫn file ảnh đầu vào từ image_id.

        Hàm này tìm trong thư mục lưu trữ ảnh bằng cách ghép image_id với các
        phần
        mở rộng ảnh được hỗ trợ. Nếu tìm thấy file phù hợp, hàm trả về đường dẫn
        ảnh. Nếu không tìm thấy, hàm sẽ raise HTTPException.

        Tham số:
          image_id: Mã định danh của ảnh khuôn mặt đầu vào.

        Trả về:
          Path tới file ảnh tương ứng với image_id.
        """

        image_storage = Path(IMAGE_STORAGE)

        for suffix in IMAGE_SUFFIXES:
            candidate = image_storage / f"{image_id}{suffix}"

            if candidate.is_file():
                return candidate

        raise HTTPException(
            status_code=404,
            detail=f"Image not found for image_id: {image_id}",
        )

  # async def _save_upload(
  #     self,
  #     upload: UploadFile,
  #     target_dir:Path,
  #     filename: str,
  #     allowed_suffixes: set[str]
  # ) -> Path:
  #   suffix = Path(upload.filename or "").suffix.lower()
  #   if suffix not in allowed_suffixes:
  #     raise HTTPException(
  #       status_code=415,
  #       detail=f"Unsupported file type: {upload.filename}"
  #     ) 
  #   output_path = target_dir / f"{filename}{suffix}"
    
  #   try: 
  #     await run_in_threadpool(
  #       self._copy_upload_to_path,
  #       upload,
  #       output_path
  #     )
  #   finally:
  #     await upload.close()

  #   if output_path.stat().st_size == 0:
  #     raise HTTPException(
  #       status_code=400,
  #       detail=f"Uploaded file is empty: {upload.filename}"
  #     )
  #   return output_path