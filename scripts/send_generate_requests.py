#!/usr/bin/env python3
"""Send multiple concurrent multipart requests to the LipSync generate API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'requests'. Install it with: pip install requests"
    ) from exc


DEFAULT_URL = "http://localhost:8010/api/v1/lipsync/generate"
DEFAULT_IMAGE = "/home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/examples/base.jpg"
DEFAULT_AUDIO = "/home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/examples/output_pip_omnivoice.wav"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send concurrent image/audio uploads to POST /generate."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, type=Path, help="Input image path")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, type=Path, help="Input audio path")
    parser.add_argument("--url", default=DEFAULT_URL, help="Generate endpoint URL")
    parser.add_argument("--count", type=int, default=10, help="Number of requests")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of requests sent at the same time",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each upload request",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed form field; omit when seed is configured via ENV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generate_request_results.json"),
        help="File used to store all job_id/task_id responses",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise SystemExit(f"Image file not found: {args.image}")
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")


def content_type(path: Path, fallback: str) -> str:
    guessed_type, _ = mimetypes.guess_type(path.name)
    return guessed_type or fallback


def send_request(index: int, args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()

    try:
        with args.image.open("rb") as image_file, args.audio.open("rb") as audio_file:
            files = {
                "image": (
                    args.image.name,
                    image_file,
                    content_type(args.image, "application/octet-stream"),
                ),
                "audio": (
                    args.audio.name,
                    audio_file,
                    content_type(args.audio, "application/octet-stream"),
                ),
            }
            data = {} if args.seed is None else {"seed": str(args.seed)}

            response = requests.post(
                args.url,
                files=files,
                data=data,
                timeout=args.timeout,
            )

        try:
            body: Any = response.json()
        except ValueError:
            body = response.text

        return {
            "request": index,
            "ok": response.ok,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "response": body,
        }
    except requests.RequestException as exc:
        return {
            "request": index,
            "ok": False,
            "status_code": None,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "error": str(exc),
        }


def main() -> int:
    args = parse_args()
    validate_args(args)

    worker_count = min(args.count, args.concurrency)
    results: list[dict[str, Any]] = []

    print(
        f"Sending {args.count} request(s) to {args.url} "
        f"with concurrency={worker_count}"
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(send_request, index, args)
            for index in range(1, args.count + 1)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            response = result.get("response")
            task_id = response.get("task_id") if isinstance(response, dict) else None
            job_id = response.get("job_id") if isinstance(response, dict) else None

            print(
                f"[{result['request']:02d}] status={result['status_code']} "
                f"elapsed={result['elapsed_seconds']}s "
                f"job_id={job_id} task_id={task_id}"
            )

    results.sort(key=lambda item: item["request"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    succeeded = sum(1 for result in results if result["ok"])
    failed = len(results) - succeeded

    print(f"Completed: {succeeded} succeeded, {failed} failed")
    print(f"Results written to: {args.output.resolve()}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
