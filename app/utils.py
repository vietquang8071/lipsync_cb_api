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