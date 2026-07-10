import os
import subprocess
import time
from pathlib import Path


LOG_DIR = Path("benchmark_logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "flashhead_inference.log"
RESULT_FILE = LOG_DIR / "runtime_result.txt"


def main() -> None:
    # Biến môi trường dùng GPU vật lý số 1.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    command = [
        "python",
        "generate_video.py",
        "--ckpt_dir",
        "models/SoulX-FlashHead-1_3B",
        "--wav2vec_dir",
        "models/wav2vec2-base-960h",
        "--model_type",
        "pro",
        "--cond_image",
        "examples/base.jpg",
        "--audio_path",
        "examples/output_pip_omnivoice.wav",
        "--audio_encode_mode",
        "stream",
    ]

    print("Đang chạy SoulX-FlashHead...")
    print("Command:", " ".join(command))

    start_time = time.perf_counter()

    # Ghi toàn bộ stdout và stderr vào file log.
    with LOG_FILE.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Vừa hiển thị log ra terminal, vừa ghi vào file.
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    result = (
        f"Exit code      : {return_code}\n"
        f"Response time  : {elapsed_time:.3f} seconds\n"
        f"Response time  : {elapsed_time / 60:.3f} minutes\n"
        f"Log file       : {LOG_FILE}\n"
    )

    print("\n" + result)
    RESULT_FILE.write_text(result, encoding="utf-8")

    if return_code != 0:
        raise RuntimeError(
            f"Inference thất bại với exit code {return_code}. "
            f"Kiểm tra log tại {LOG_FILE}."
        )


if __name__ == "__main__":
    main()
