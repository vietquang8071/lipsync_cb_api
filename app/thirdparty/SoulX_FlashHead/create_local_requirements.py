from pathlib import Path


source = Path("requirements.txt")
target = Path("requirements.local.txt")

blocked_prefixes = (
    "torch==",
    "torchvision==",
    "torchaudio==",
    "nvidia-nccl-cu12==",
    "xformers==",
)

lines = source.read_text(encoding="utf-8").splitlines()

filtered = [
    line
    for line in lines
    if line.strip()
    and not line.strip().startswith(blocked_prefixes)
]

target.write_text(
    "\n".join(filtered) + "\n",
    encoding="utf-8",
)

print(f"Created: {target}")
print(f"Kept dependencies: {len(filtered)}")
