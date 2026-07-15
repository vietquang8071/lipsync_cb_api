import os 
import socket
from pathlib import Path
from time import time
from loguru import logger

from app.model.soulX import SoulXModel
from app.schemas.job import LipSyncJob
from app.worker.celery_app import celery_app
from app.service.lipsync_service import AUDIO_STORAGE
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

_model: SoulXModel | None = None


def _delete_job_audio(
    audio_path: str | Path,
) -> bool:
    path = Path(audio_path).expanduser().resolve()

    # Không bao giờ xóa file ngoài /dev/shm/audio.
    if not path.is_relative_to(AUDIO_STORAGE):
        logger.error(
            "Refusing to delete audio outside AUDIO_STORAGE; "
            "audio_path={}; audio_storage={}",
            path,
            AUDIO_STORAGE,
        )
        return False

    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Cleanup thất bại không nên biến một job đã generate
        # thành job failed và chạy lại toàn bộ inference.
        logger.exception(
            "Failed to delete job audio; audio_path={}",
            path,
        )
        return False

    logger.info(
        "Job audio deleted; audio_path={}",
        path,
    )
    return True


def _resolve_required_directory(
    env_name: str,
    default: Path,
) -> Path:
    raw_value = os.getenv(env_name)
    path = Path(raw_value) if raw_value else default

    path = path.expanduser()

    if not path.is_absolute():
        raise RuntimeError(
            f"{env_name} must be an absolute path: {path}"
        )

    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{env_name} directory not found: {path}"
        ) from exc

    if not path.is_dir():
        raise RuntimeError(
            f"{env_name} is not a directory: {path}"
        )

    if not os.access(path, os.R_OK):
        raise RuntimeError(
            f"{env_name} is not readable: {path}"
        )

    return path


def _resolve_required_file(
    env_name: str,
    default: Path,
) -> Path:
    raw_value = os.getenv(env_name)
    path = Path(raw_value) if raw_value else default

    path = path.expanduser()

    if not path.is_absolute():
        raise RuntimeError(
            f"{env_name} must be an absolute path: {path}"
        )

    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{env_name} file not found: {path}"
        ) from exc

    if not path.is_file():
        raise RuntimeError(
            f"{env_name} is not a file: {path}"
        )

    if not os.access(path, os.R_OK):
        raise RuntimeError(
            f"{env_name} is not readable: {path}"
        )

    return path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"{label} not found: {path}"
        )

    if not os.access(path, os.R_OK):
        raise RuntimeError(
            f"{label} is not readable: {path}"
        )


def _require_weights(
    directory: Path,
    label: str,
) -> None:
    patterns = (
        "*.safetensors",
        "*.bin",
        "*.pth",
    )

    has_weights = any(
        path.is_file()
        for pattern in patterns
        for path in directory.glob(pattern)
    )

    if not has_weights:
        raise RuntimeError(
            f"{label} weights not found in: {directory}"
        )

def validate_model_configuration() -> tuple[
    Path,
    Path,
    Path,
    str,
]:
    checkpoint_dir = _resolve_required_directory(
        "SOULX_CKPT_DIR",
        SOULX_DIR / "models" / "SoulX-FlashHead-1_3B",
    )

    wav2vec_dir = _resolve_required_directory(
        "SOULX_WAV2VEC_DIR",
        SOULX_DIR / "models" / "wav2vec2-base-960h",
    )

    config_path = _resolve_required_file(
        "SOULX_CONFIG_PATH",
        SOULX_DIR / "flash_head" / "configs" / "infer_params.yaml",
    )

    model_type = (
        os.getenv("SOULX_MODEL_TYPE", "lite")
        .strip()
        .lower()
    )

    if model_type not in {"lite", "pro", "pretrained"}:
        raise RuntimeError(
            "SOULX_MODEL_TYPE must be one of: "
            "lite, pro, pretrained"
        )

    if model_type == "lite":
        model_dir = checkpoint_dir / "Model_Lite"
        vae_dir = checkpoint_dir / "VAE_LTX"

        if not model_dir.is_dir():
            raise RuntimeError(
                f"Model_Lite directory not found: {model_dir}"
            )

        if not vae_dir.is_dir():
            raise RuntimeError(
                f"VAE_LTX directory not found: {vae_dir}"
            )

        _require_file(
            model_dir / "config.json",
            "Model Lite config",
        )
        _require_weights(
            model_dir,
            "Model Lite",
        )

        _require_file(
            vae_dir / "config.json",
            "VAE LTX config",
        )
        _require_weights(
            vae_dir,
            "VAE LTX",
        )

    elif model_type == "pro":
        model_dir = checkpoint_dir / "Model_Pro"
        vae_path = (
            checkpoint_dir
            / "VAE_Wan"
            / "Wan2.1_VAE.pth"
        )

        if not model_dir.is_dir():
            raise RuntimeError(
                f"Model_Pro directory not found: {model_dir}"
            )

        _require_file(
            model_dir / "config.json",
            "Model Pro config",
        )
        _require_weights(
            model_dir,
            "Model Pro",
        )
        _require_file(
            vae_path,
            "Wan VAE weights",
        )

    else:
        model_dir = checkpoint_dir / "teacher"
        vae_path = (
            checkpoint_dir
            / "VAE_Wan"
            / "Wan2.1_VAE.pth"
        )

        if not model_dir.is_dir():
            raise RuntimeError(
                f"Teacher model directory not found: {model_dir}"
            )

        _require_file(
            model_dir / "config.json",
            "Teacher model config",
        )
        _require_weights(
            model_dir,
            "Teacher model",
        )
        _require_file(
            vae_path,
            "Wan VAE weights",
        )

    _require_file(
        wav2vec_dir / "config.json",
        "Wav2Vec config",
    )
    _require_weights(
        wav2vec_dir,
        "Wav2Vec",
    )

    logger.info(
        "SoulX model configuration validated; "
        "model_type={}; checkpoint_dir={}; "
        "wav2vec_dir={}; config_path={}",
        model_type,
        checkpoint_dir,
        wav2vec_dir,
        config_path,
    )

    return (
        checkpoint_dir,
        wav2vec_dir,
        config_path,
        model_type,
    )

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

    (
        checkpoint_dir,
        wav2vec_dir,
        config_path,
        model_type,
    ) = validate_model_configuration()

    _model = SoulXModel(
        cpkt_dir=str(checkpoint_dir),
        wav2vec_dir=str(wav2vec_dir),
        model_type=model_type,
        config_path=str(config_path),
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

      use_face_crop = (
        os.getenv("USE_FACE_CROP", "false")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
      ) 
      model.generate_video(
        Path(image_path),
        Path(audio_path),
        output_path=output_path,
        seed=int(os.getenv("SEED", 42)),
        use_face_crop=use_face_crop
      )

      if (
          not output_path.is_file()
          or output_path.stat().st_size == 0
      ):
          raise RuntimeError(
              "Output video was not created or is empty"
          )
      
      logger.info(
        "LipSync generate_video completed job_id={} output_path={} output_size={} elapsed_sec={:.2f}",
        job.job_id,
        output_path,
        output_path.stat().st_size,
        time() - started,
      )

      _delete_job_audio(audio_path)

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
        self.update_state(state="PROGRESS", 
                          meta=_dump_job(job))
        raise self.retry(exc=exc, 
                         countdown=10 * job.attempts)

      job.status = "failed"
      raise

    job.status = "completed"
    job.completed_at = time()
    job.updated_at = job.completed_at
    return _dump_job(job)

