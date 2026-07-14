from pathlib import Path
import subprocess

import imageio
import numpy as np


def save_video(
    frames_list: list,
    video_path: Path,
    audio_path: Path,
    fps: int,
) -> None:
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_video_path = video_path.with_name(
        f"{video_path.stem}_raw.mp4"
    )
    muxing_path = video_path.with_name(
        f"{video_path.stem}_muxing.mp4"
    )

    try:
        with imageio.get_writer(
            str(raw_video_path),
            format="mp4",
            mode="I",
            fps=fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frames in frames_list:
                frames_np = (
                    frames.numpy()
                    .astype(np.uint8)
                )

                for frame in frames_np:
                    writer.append_data(frame)

        cmd = [
            "ffmpeg",
            "-i",
            str(raw_video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(muxing_path),
            "-y",
        ]

        process = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {process.stderr}"
            )

        if (
            not muxing_path.exists()
            or muxing_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "ffmpeg did not create a valid output file"
            )

        # Đổi tên nguyên tử sau khi ffmpeg đã hoàn thành.
        # Chỉ lúc này lipsync.mp4 mới xuất hiện.
        muxing_path.replace(video_path)

    finally:
        if raw_video_path.exists():
            raw_video_path.unlink()

        if muxing_path.exists():
            muxing_path.unlink()

def save_video_with_waveform(
    frames_list: list,
    video_path: Path,
    waveform: np.ndarray,
    fps: int,
    sample_rate: int = 16_000,
) -> None:
    video_path = Path(video_path)

    if fps <= 0:
        raise ValueError("FPS must be greater than 0")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    # Chuẩn hóa waveform thành mono float32 liên tục trong bộ nhớ.
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        raise ValueError("Waveform is empty")

    # Nếu có dạng (1, samples), chuyển thành (samples,).
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]

    # Nếu có dạng (samples, 1), chuyển thành (samples,).
    elif waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]

    elif waveform.ndim != 1:
        raise ValueError(
            f"Expected mono waveform with shape (samples,), "
            f"(1, samples), or (samples, 1), got {waveform.shape}"
        )

    waveform = np.ascontiguousarray(
        waveform,
        dtype=np.float32,
    )

    nan_count = int(np.isnan(waveform).sum())
    inf_count = int(np.isinf(waveform).sum())

    if nan_count or inf_count:
        raise ValueError(
            f"Waveform contains invalid values: "
            f"NaN={nan_count}, Inf={inf_count}"
        )

    duration = waveform.size / sample_rate
    min_value = float(waveform.min())
    max_value = float(waveform.max())

    print("Waveform dtype:", waveform.dtype)
    print("Waveform shape:", waveform.shape)
    print("Samples:", waveform.size)
    print("Duration:", duration)
    print("NaN:", nan_count)
    print("Inf:", inf_count)
    print("Min:", min_value)
    print("Max:", max_value)
    print("First samples:", waveform[:10])

    # PCM float thường nằm trong khoảng [-1, 1].
    # Không nhất thiết phải tuyệt đối đúng, nhưng giá trị quá lớn thường là lỗi.
    if max(abs(min_value), abs(max_value)) > 10:
        raise ValueError(
            "Waveform amplitude is suspiciously large. "
            "Expected decoded float32 PCM, usually around [-1.0, 1.0]."
        )

    audio_bytes = waveform.tobytes()

    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_video_path = video_path.with_name(
        f"{video_path.stem}_raw.mp4"
    )
    muxing_path = video_path.with_name(
        f"{video_path.stem}_muxing.mp4"
    )

    try:
        # Bước 1: ghi frame thành video chưa có âm thanh.
        with imageio.get_writer(
            str(raw_video_path),
            format="mp4",
            mode="I",
            fps=fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frames in frames_list:
                if hasattr(frames, "detach"):
                    frames_np = (
                        frames.detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                else:
                    frames_np = np.asarray(
                        frames,
                        dtype=np.uint8,
                    )

                for frame in frames_np:
                    writer.append_data(frame)

        # Bước 2: truyền raw PCM float32 vào stdin của FFmpeg.
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",

            # Input 0: video.
            "-i",
            str(raw_video_path),

            # Input 1: raw PCM float32, mono.
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",

            # Chọn stream video và audio.
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",

            # Giữ nguyên video H.264.
            "-c:v",
            "copy",

            # Encode audio PCM sang AAC.
            "-c:a",
            "aac",

            "-shortest",
            "-movflags",
            "+faststart",

            str(muxing_path),
        ]

        process = subprocess.run(
            cmd,
            input=audio_bytes,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.returncode != 0:
            stderr = process.stderr.decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"ffmpeg failed with code "
                f"{process.returncode}:\n{stderr}"
            )

        if (
            not muxing_path.exists()
            or muxing_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "ffmpeg did not create a valid output file"
            )

        muxing_path.replace(video_path)

    finally:
        if raw_video_path.exists():
            raw_video_path.unlink()

        if muxing_path.exists():
            muxing_path.unlink()

def decode_audio_buffer(
  audio_buffer: bytes | bytearray | memoryview,
  sample_rate: int = 16000      
):
    if not audio_buffer:
        raise ValueError("Audio buffer is empty")
    process = subprocess.run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ],
    input=bytes(audio_buffer),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False
    )

    if process.returncode != 0:
        error = process.sterr.decode("utf-8", error="replace")
        raise ValueError(f"Cannot decode audio buffer: {error}")
    waveform = np.frombuffer(process.stdout, dtype=np.float32).copy()

    if waveform.size == 0:
        raise ValueError("Decoded audio contains no samples")
    return waveform

