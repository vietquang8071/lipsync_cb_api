#!/usr/bin/env python3
"""Benchmark SoulX-FlashHead with one-time Wav2Vec2 audio encoding.

This script is intentionally standalone: it imports the existing pipeline but does
not require changes to any project source file. Run it from the repository root.

The benchmark measures:
  * pipeline/model loading;
  * reference-image preparation;
  * audio loading and padding;
  * one full-audio Wav2Vec2 pass;
  * per-chunk audio-window construction;
  * per-chunk video generation;
  * optional GPU-to-CPU frame transfer.

It does not encode or mux an MP4, so video-codec time is excluded.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import librosa
import numpy as np
import torch

from flash_head.inference import (
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)


T = TypeVar("T")


def synchronize_cuda() -> None:
    """Wait for queued CUDA work so wall-clock GPU timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(call: Callable[[], T]) -> tuple[T, float]:
    """Run a callable and return its result and synchronized elapsed seconds."""
    synchronize_cuda()
    started_at = time.perf_counter()
    result = call()
    synchronize_cuda()
    return result, time.perf_counter() - started_at


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_audio_embedding_chunk(
    audio_features: torch.Tensor,
    start_idx: int,
    frame_num: int,
) -> torch.Tensor:
    """Build one [1, frame_num, 5, 12, 768] conditioning tensor."""
    total_frames = audio_features.shape[0]
    end_idx = start_idx + frame_num

    if start_idx < 0 or end_idx > total_frames:
        raise ValueError(
            "Audio chunk is outside the available feature range: "
            f"start={start_idx}, end={end_idx}, total={total_frames}"
        )

    offsets = torch.arange(
        -2,
        3,
        device=audio_features.device,
        dtype=torch.long,
    )
    centers = torch.arange(
        start_idx,
        end_idx,
        device=audio_features.device,
        dtype=torch.long,
    )
    indices = centers[:, None] + offsets[None, :]
    indices = indices.clamp(min=0, max=total_frames - 1)
    return audio_features[indices][None].contiguous()


def pad_audio_for_chunks(
    audio: np.ndarray,
    sample_rate: int,
    fps: int,
    frame_num: int,
    slice_len: int,
) -> tuple[np.ndarray, int]:
    """Pad audio so its frame count is frame_num + N * slice_len."""
    if sample_rate % fps != 0:
        raise ValueError(
            "This benchmark expects an integer number of audio samples per "
            f"video frame, got sample_rate={sample_rate}, fps={fps}."
        )

    original_sample_count = len(audio)
    samples_per_frame = sample_rate // fps
    first_chunk_samples = frame_num * samples_per_frame
    stride_samples = slice_len * samples_per_frame

    if len(audio) < first_chunk_samples:
        audio = np.pad(audio, (0, first_chunk_samples - len(audio)))

    remaining_samples = len(audio) - first_chunk_samples
    pad_length = (-remaining_samples) % stride_samples
    if pad_length:
        audio = np.pad(audio, (0, pad_length))

    return audio, original_sample_count


@dataclass
class ChunkTiming:
    index: int
    embedding_seconds: float
    generation_seconds: float
    transfer_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.embedding_seconds
            + self.generation_seconds
            + self.transfer_seconds
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure FlashHead inference with Wav2Vec2 encoded once for the "
            "complete audio file."
        )
    )
    parser.add_argument("--ckpt-dir", "--ckpt_dir", required=True)
    parser.add_argument("--wav2vec-dir", "--wav2vec_dir", required=True)
    parser.add_argument("--model-type", "--model_type", choices=("lite", "pro"), required=True)
    parser.add_argument("--cond-image", "--cond_image", required=True)
    parser.add_argument("--audio-path", "--audio_path", required=True)
    parser.add_argument("--base-seed", "--base_seed", type=int, default=42)
    parser.add_argument("--use-face-crop", "--use_face_crop", action="store_true")
    parser.add_argument(
        "--max-chunks",
        "--max_chunks",
        type=int,
        default=None,
        help="Benchmark only the first N chunks; default processes all chunks.",
    )
    parser.add_argument(
        "--warmup-runs",
        "--warmup_runs",
        type=int,
        default=0,
        help=(
            "Run N unmeasured generation warm-ups, then reset reference state. "
            "Warm-up duration is still reported separately."
        ),
    )
    parser.add_argument(
        "--skip-cpu-copy",
        "--skip_cpu_copy",
        action="store_true",
        help="Do not copy generated frames to CPU.",
    )
    return parser.parse_args()


def print_summary(
    stage_times: dict[str, float],
    chunk_timings: list[ChunkTiming],
    generated_frames: int,
    fps: int,
    original_audio_seconds: float,
    padded_audio_seconds: float,
) -> None:
    generation_times = [item.generation_seconds for item in chunk_timings]
    chunk_totals = [item.total_seconds for item in chunk_timings]
    warm_generation_times = generation_times[1:]
    generated_seconds = generated_frames / fps
    measured_inference = (
        stage_times["wav2vec_once"]
        + sum(chunk_totals)
    )

    print("\n=== Stage timings ===")
    for name, seconds in stage_times.items():
        print(f"{name:28s}: {seconds:10.4f} s")

    print("\n=== Per-chunk timings ===")
    print("chunk  embedding(s)  generation(s)  cpu_copy(s)  total(s)")
    for item in chunk_timings:
        print(
            f"{item.index:5d}  "
            f"{item.embedding_seconds:12.4f}  "
            f"{item.generation_seconds:13.4f}  "
            f"{item.transfer_seconds:11.4f}  "
            f"{item.total_seconds:8.4f}"
        )

    print("\n=== Aggregate ===")
    print(f"chunks                      : {len(chunk_timings)}")
    print(f"original audio              : {original_audio_seconds:.3f} s")
    print(f"padded audio                : {padded_audio_seconds:.3f} s")
    print(f"generated video             : {generated_seconds:.3f} s")
    if generation_times:
        print(f"first chunk generation      : {generation_times[0]:.4f} s")
        print(f"all chunks generation mean  : {statistics.fmean(generation_times):.4f} s")
        print(f"all chunks generation p50   : {percentile(generation_times, 0.50):.4f} s")
        print(f"all chunks generation p95   : {percentile(generation_times, 0.95):.4f} s")
    if warm_generation_times:
        print(f"chunks 1..N generation mean : {statistics.fmean(warm_generation_times):.4f} s")
        print(f"chunks 1..N generation p50  : {percentile(warm_generation_times, 0.50):.4f} s")
        print(f"chunks 1..N generation p95  : {percentile(warm_generation_times, 0.95):.4f} s")
    if generated_seconds > 0:
        print(f"measured inference RTF      : {measured_inference / generated_seconds:.4f}")
    if torch.cuda.is_available():
        print(
            "peak CUDA allocated          : "
            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GiB"
        )
        print(
            "peak CUDA reserved           : "
            f"{torch.cuda.max_memory_reserved() / (1024 ** 3):.3f} GiB"
        )


def main() -> None:
    args = parse_args()
    if args.max_chunks is not None and args.max_chunks < 1:
        raise ValueError("--max-chunks must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")

    wall_started_at = time.perf_counter()
    stage_times: dict[str, float] = {}
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    pipeline, stage_times["load_pipeline"] = measure(
        lambda: get_pipeline(
            world_size=world_size,
            ckpt_dir=args.ckpt_dir,
            model_type=args.model_type,
            wav2vec_dir=args.wav2vec_dir,
        )
    )

    _, stage_times["prepare_reference"] = measure(
        lambda: get_base_data(
            pipeline,
            cond_image_path_or_dir=args.cond_image,
            base_seed=args.base_seed,
            use_face_crop=args.use_face_crop,
        )
    )

    params = get_infer_params()
    sample_rate = int(params["sample_rate"])
    fps = int(params["tgt_fps"])
    frame_num = int(params["frame_num"])
    motion_frames_num = int(params["motion_frames_num"])
    slice_len = frame_num - motion_frames_num

    audio, stage_times["load_audio"] = measure(
        lambda: librosa.load(
            args.audio_path,
            sr=sample_rate,
            mono=True,
        )[0]
    )
    audio = np.asarray(audio, dtype=np.float32)
    audio, original_sample_count = pad_audio_for_chunks(
        audio=audio,
        sample_rate=sample_rate,
        fps=fps,
        frame_num=frame_num,
        slice_len=slice_len,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    audio_features, stage_times["wav2vec_once"] = measure(
        lambda: pipeline.preprocess_audio(
            audio,
            sr=sample_rate,
            fps=fps,
        )
    )

    total_audio_frames = int(audio_features.shape[0])
    if total_audio_frames < frame_num:
        raise RuntimeError(
            f"Wav2Vec2 returned only {total_audio_frames} frames; "
            f"at least {frame_num} are required."
        )

    available_chunks = 1 + (total_audio_frames - frame_num) // slice_len
    chunk_count = available_chunks
    if args.max_chunks is not None:
        chunk_count = min(chunk_count, args.max_chunks)

    first_embedding = build_audio_embedding_chunk(
        audio_features=audio_features,
        start_idx=0,
        frame_num=frame_num,
    )

    warmup_started_at = time.perf_counter()
    for _ in range(args.warmup_runs):
        warmup_video = run_pipeline(pipeline, first_embedding)
        del warmup_video
    synchronize_cuda()
    stage_times["generation_warmup"] = time.perf_counter() - warmup_started_at

    if args.warmup_runs:
        # generate() mutates latent motion and RNG state. Reset both after warm-up.
        _, stage_times["reset_after_warmup"] = measure(
            lambda: get_base_data(
                pipeline,
                cond_image_path_or_dir=args.cond_image,
                base_seed=args.base_seed,
                use_face_crop=args.use_face_crop,
            )
        )

    chunk_timings: list[ChunkTiming] = []
    with torch.inference_mode():
        for chunk_idx in range(chunk_count):
            audio_start_idx = chunk_idx * slice_len

            embedding, embedding_seconds = measure(
                lambda start=audio_start_idx: build_audio_embedding_chunk(
                    audio_features=audio_features,
                    start_idx=start,
                    frame_num=frame_num,
                )
            )
            video, generation_seconds = measure(
                lambda current=embedding: run_pipeline(pipeline, current)
            )

            if chunk_idx != 0:
                video = video[motion_frames_num:]

            transfer_seconds = 0.0
            if not args.skip_cpu_copy:
                host_video, transfer_seconds = measure(video.cpu)
                del host_video

            del embedding
            del video

            timing = ChunkTiming(
                index=chunk_idx,
                embedding_seconds=embedding_seconds,
                generation_seconds=generation_seconds,
                transfer_seconds=transfer_seconds,
            )
            chunk_timings.append(timing)
            print(
                f"chunk={chunk_idx}/{chunk_count - 1} "
                f"embedding={embedding_seconds:.4f}s "
                f"generation={generation_seconds:.4f}s "
                f"cpu_copy={transfer_seconds:.4f}s "
                f"total={timing.total_seconds:.4f}s"
            )

    stage_times["total_wall"] = time.perf_counter() - wall_started_at
    generated_frames = frame_num + max(0, chunk_count - 1) * slice_len
    print_summary(
        stage_times=stage_times,
        chunk_timings=chunk_timings,
        generated_frames=generated_frames,
        fps=fps,
        original_audio_seconds=original_sample_count / sample_rate,
        padded_audio_seconds=len(audio) / sample_rate,
    )


if __name__ == "__main__":
    main()
