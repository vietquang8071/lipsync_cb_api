from pathlib import Path
import imageio
import subprocess
import numpy as np 
def save_video(
  frames_list: list,
  video_path:Path,
  audio_path:Path,
  fps:int
) -> None:
  temp_video_path = video_path.with_name(f"{video_path.stem}_tmp.mp4")
  
  with imageio.get_writer(
    str(temp_video_path),
    format="mp4",
    mode="I",
    fps=fps,
    codec="h264",
    ffmpeg_params=["-bf","0"]
  ) as writer:
    for frames in frames_list:
      frames_np = frames.numpy().astype(np.uint8)
      for i in range(frames_np.shape[0]):
        writer.append_data(frames_np[i])

  cmd = [
      "ffmpeg",
      "-i",
      str(temp_video_path),
      "-i",
      str(audio_path),
      "-c:v",
      "copy",
      "-c:a",
      "aac",
      "-shortest",
      str(video_path),
      "-y",
  ]

  try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
  finally:
    if temp_video_path.exists():
      temp_video_path.unlink()