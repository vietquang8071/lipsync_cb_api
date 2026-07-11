import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import torch

from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)


def decode_audio_buffer(
    audio_buffer: bytes | bytearray | memoryview,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Decode WAV/MP3/M4A buffer thành mono float32 waveform."""

    if not audio_buffer:
        raise ValueError("Audio buffer is empty")

    process = subprocess.run(
        [
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
        check=False,
    )

    if process.returncode != 0:
        error = process.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"Cannot decode audio buffer: {error}")

    waveform = np.frombuffer(process.stdout, dtype=np.float32).copy()

    if waveform.size == 0:
        raise ValueError("Decoded audio contains no samples")

    return waveform


def generate_frames_from_audio_buffer(
    pipeline,
    audio_buffer: bytes | bytearray | memoryview,
) -> tuple[list[torch.Tensor], int]:
    """Đưa audio buffer vào SoulX và trả về danh sách video frames."""

    params = get_infer_params()

    sample_rate = params["sample_rate"]
    fps = params["tgt_fps"]
    frame_num = params["frame_num"]
    motion_frames_num = params["motion_frames_num"]
    cached_audio_duration = params["cached_audio_duration"]

    slice_len = frame_num - motion_frames_num
    audio_slice_len = slice_len * sample_rate // fps

    # bytes -> mono float32 waveform 16 kHz
    speech_array = decode_audio_buffer(
        audio_buffer,
        sample_rate=sample_rate,
    )

    # Pad chunk cuối bằng silence
    pad_length = (-len(speech_array)) % audio_slice_len
    if pad_length:
        speech_array = np.pad(speech_array, (0, pad_length))

    audio_chunks = speech_array.reshape(-1, audio_slice_len)

    cached_audio_length = int(sample_rate * cached_audio_duration)
    audio_end_idx = int(cached_audio_duration * fps)
    audio_start_idx = audio_end_idx - frame_num

    audio_cache = deque(
        [0.0] * cached_audio_length,
        maxlen=cached_audio_length,
    )

    generated_frames = []

    with torch.inference_mode():
        for speech_chunk in audio_chunks:
            audio_cache.extend(speech_chunk)

            audio_array = np.fromiter(
                audio_cache,
                dtype=np.float32,
                count=cached_audio_length,
            )

            audio_embedding = get_audio_embedding(
                pipeline,
                audio_array,
                audio_start_idx,
                audio_end_idx,
            )

            video = run_pipeline(pipeline, audio_embedding)

            # Bỏ các frame chuyển động được cache từ chunk trước
            video = video[motion_frames_num:]
            generated_frames.append(video.cpu())

    if not generated_frames:
        raise RuntimeError("SoulX did not generate any frames")

    return generated_frames, fps


if __name__ == "__main__":
    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir="models/SoulX-FlashHead-1_3B",
        wav2vec_dir="models/wav2vec2-base-960h",
        model_type="lite",
    )

    get_base_data(
        pipeline,
        cond_image_path_or_dir="examples/base.jpg",
        base_seed=42,
        use_face_crop=False,
    )

    # Đọc toàn bộ file dưới dạng encoded bytes buffer
    audio_buffer = Path(
        "examples/output_pip_omnivoice.wav"
    ).read_bytes()

    frames, fps = generate_frames_from_audio_buffer(
        pipeline,
        audio_buffer,
    )

    print(f"Generated {sum(len(chunk) for chunk in frames)} frames at {fps} FPS")