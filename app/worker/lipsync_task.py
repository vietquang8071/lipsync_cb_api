import os 
import socket
from pathlib import Path
from time import time
from loguru import logger

from app.model.soulX import SoulXModel
from app.schemas.job import LipSyncJob
from app.worker.celery_app import celery_app
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

_model: SoulXModel | None = None

def _dump_job(job: LipSyncJob) -> dict:
    """
    Chuyển đối tượng LipSyncJob sang dạng dict.

    Hàm này hỗ trợ cả Pydantic v2 và Pydantic v1 bằng cách ưu tiên dùng
    model_dump() nếu có, nếu không thì dùng dict(). Kết quả được dùng để cập
    nhật
    trạng thái Celery task hoặc trả metadata job về cho caller.

    Tham số:
      job: Đối tượng LipSyncJob cần được chuyển đổi.

    Trả về:
      Dict chứa toàn bộ thông tin hiện tại của job.
    """

    return job.model_dump() if hasattr(job, "model_dump") else job.dict()

def _get_model() -> SoulXModel:
    """
    Khởi tạo hoặc lấy lại instance SoulXModel dùng để tạo video lipsync.

    Hàm này dùng biến global _model để cache model sau lần khởi tạo đầu tiên,
    giúp worker không phải load lại model cho mỗi job. Các đường dẫn checkpoint,
    wav2vec, config và loại model được lấy từ biến môi trường nếu có, ngược lại
    dùng giá trị mặc định trong thư mục SoulX_FlashHead.

    Trả về:
      Instance SoulXModel đã sẵn sàng để sinh video lipsync.
    """

    global _model
    if _model is not None:
      return _model
    

    _model = SoulXModel(
      cpkt_dir=os.getenv(
          "SOULX_CKPT_DIR",
          str(SOULX_DIR / "models" / "SoulX-FlashHead-1_3B"),
      ),
      wav2vec_dir=os.getenv(
          "SOULX_WAV2VEC_DIR",
          str(SOULX_DIR / "models" / "wav2vec2-base-960h"),
      ),
      model_type=os.getenv("SOULX_MODEL_TYPE", "lite"),
      config_path=os.getenv(
          "SOULX_CONFIG_PATH",
          str(SOULX_DIR / "flash_head" / "configs" / "infer_params.yaml"),
      ),
      world_size=int(os.getenv("WORLD_SIZE", "1")),
    )
    return _model

@celery_app.task(name="process_lipsync_task", bind=True, max_retries=2)
def processing_lipsync_task(self, job_payload:dict) -> dict:
    """
    Xử lý job tạo video lipsync trong Celery worker.

    Hàm này nhận payload của LipSyncJob, cập nhật trạng thái job sang
    processing,
    load model lipsync, gọi model để sinh video từ ảnh và audio đầu vào, sau đó
    cập nhật trạng thái completed nếu thành công. Nếu có lỗi, hàm ghi lại thông
    tin lỗi, cập nhật trạng thái retry hoặc failed và thực hiện retry theo cấu
    hình
    của Celery task.

    Tham số:
      self: Instance Celery task hiện tại. Tham số này có vì task được khai báo
        với bind=True, cho phép truy cập thông tin retry, max_retries và cập
        nhật
        state của task.
      job_payload: Dict chứa dữ liệu của LipSyncJob, gồm job_id, image_path,
        audio_path, output_path và các metadata liên quan đến trạng thái xử lý.

    Trả về:
      Dict chứa thông tin job sau khi xử lý, bao gồm trạng thái cuối cùng hoặc
      trạng thái lỗi nếu job thất bại.
    """

    started = time()
    job = LipSyncJob(**job_payload)
    job.status = "processing"
    job.worker_id = f"{socket.gethostname()}:{os.getpid()}"
    job.attempts = self.request.retries + 1
    job.started_at = job.started_at or time()
    job.updated_at = time()

    logger.info(
      "LipSync task started job_id={}; attempt={}; worker_id={}; image_path={}; audio_path={}; output_path={}",
      job.job_id,
      job.attempts,
      job.worker_id,
      job.image_path,
      job.audio_path,
      job.output_path,
    )

    self.update_state(state="PROGRESS", meta=_dump_job(job))

    try:

      image_path = Path(job.image_path)
      audio_path = Path(job.audio_path)

      logger.info("LipSync loading model job_id={}", job.job_id)
      model = _get_model()
      logger.info("LipSync model ready job_id={}", job.job_id)

      output_path = Path(job.output_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      
      logger.info(
        "LipSync generate_video started job_id={}",
        job.job_id,
      )

      model.generate_video(
        Path(image_path),
        Path(audio_path),
        output_path=output_path,
        seed=int(os.getenv("SEED", 42)),
        use_face_crop=os.getenv("USE_FACE_CROP", default=False)
      )

      if not output_path.exists():
        raise RuntimeError("Output video was not created")
      
      logger.info(
        "LipSync generate_video completed job_id={} output_path={} output_size={} elapsed_sec={:.2f}",
        job.job_id,
        output_path,
        output_path.stat().st_size,
        time() - started,
      )


    except Exception as exc:
      will_retry = self.request.retries < self.max_retries

      logger.exception(
        "LipSync task failed job_id={} attempt={} max_retries={} will_retry={} error_type={} error={}",
        job.job_id,
        job.attempts,
        self.max_retries,
        will_retry,
        type(exc).__name__,
        str(exc),
      )

      job.error = f"{type(exc).__name__}: {exc}"
      job.updated_at = time()

      if will_retry:
        job.status = "retry"
        self.update_state(state="PROGRESS", meta=_dump_job(job))
        raise self.retry(exc=exc, countdown=10 * job.attempts)

      job.status = "failed"
      return _dump_job(job)

    job.status = "completed"
    job.completed_at = time()
    job.updated_at = job.completed_at
    return _dump_job(job)

