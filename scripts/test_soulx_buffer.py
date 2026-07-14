#!/usr/bin/env python3
"""Run SoulXModel directly with an image path and an audio byte buffer.

This script does not call the HTTP API or Celery worker. It loads
app/model/soulX.py, reads the audio file as bytes, and passes that buffer
directly to SoulXModel.generate_frames_from_audio_buffer().
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

DEFAULT_IMAGE = SOULX_DIR / "examples" / "base.jpg"
DEFAULT_AUDIO = SOULX_DIR / "examples" / "output_pip_omnivoice.wav"
DEFAULT_OUTPUT = REPO_ROOT / "storage" / "soulx_buffer_test" / "lipsync.mp4"
DEFAULT_CKPT_DIR = SOULX_DIR / "models" / "SoulX-FlashHead-1_3B"
DEFAULT_WAV2VEC_DIR = SOULX_DIR / "models" / "wav2vec2-base-960h"
DEFAULT_CONFIG = SOULX_DIR / "flash_head" / "configs" / "infer_params.yaml"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected one of: true/false, yes/no, 1/0, on/off"
    )


def existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} file not found: {resolved}")
    return resolved


def existing_dir(path: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"{label} directory not found: {resolved}")
    return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test app/model/soulX.py directly with image path + audio bytes "
            "buffer, without calling the API endpoint."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(os.getenv("SOULX_TEST_IMAGE", DEFAULT_IMAGE)),
        help="Input face image path.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(os.getenv("SOULX_TEST_AUDIO", DEFAULT_AUDIO)),
        help="Audio file whose bytes will be passed as audio_buffer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("SOULX_TEST_OUTPUT", DEFAULT_OUTPUT)),
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(os.getenv("SOULX_CKPT_DIR", DEFAULT_CKPT_DIR)),
        help="SoulX checkpoint directory.",
    )
    parser.add_argument(
        "--wav2vec-dir",
        type=Path,
        default=Path(os.getenv("SOULX_WAV2VEC_DIR", DEFAULT_WAV2VEC_DIR)),
        help="Wav2Vec checkpoint directory.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(os.getenv("SOULX_CONFIG_PATH", DEFAULT_CONFIG)),
        help="SoulX infer_params.yaml path.",
    )
    parser.add_argument(
        "--model-type",
        default=os.getenv("SOULX_MODEL_TYPE", "lite"),
        choices=("lite", "pro"),
        help="SoulX model type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Base seed for generation.",
    )
    parser.add_argument(
        "--use-face-crop",
        type=str_to_bool,
        default=str_to_bool(os.getenv("USE_FACE_CROP", "false")),
        help="Whether SoulX should crop the face before inference.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="Distributed world size used by SoulX.",
    )
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Do not unload the model before exiting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    from app.model.soulX import SoulXModel

    image_path = existing_file(args.image, "Image")
    audio_path = existing_file(args.audio, "Audio")
    config_path = existing_file(args.config_path, "Config")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_buffer = audio_path.read_bytes()
    if not audio_buffer:
        raise SystemExit(f"Audio file is empty: {audio_path}")

    print(f"Image: {image_path}")
    print(f"Audio: {audio_path} ({len(audio_buffer):,} bytes passed as buffer)")
    print(f"Output: {output_path}")

    model = SoulXModel(
        cpkt_dir=existing_dir(args.ckpt_dir, "Checkpoint"),
        wav2vec_dir=existing_dir(args.wav2vec_dir, "Wav2Vec"),
        model_type=args.model_type,
        config_path=str(config_path),
        world_size=args.world_size,
    )

    started_at = perf_counter()
    try:
        model.load()
        model.generate_frames_from_audio_buffer(
            image_path=image_path,
            audio_buffer=audio_buffer,
            output_path=output_path,
            seed=args.seed,
            use_face_crop=args.use_face_crop,
        )
    finally:
        if not args.keep_loaded:
            model.unload()

    elapsed = perf_counter() - started_at
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise SystemExit(f"SoulX did not create a valid output file: {output_path}")

    print(
        "Done: "
        f"{output_path} ({output_path.stat().st_size:,} bytes, {elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
