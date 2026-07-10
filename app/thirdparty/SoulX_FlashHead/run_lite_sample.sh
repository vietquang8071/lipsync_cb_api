#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type pro \
    --cond_image examples/images.jpeg \
    --audio_path examples/output_pip_omnivoice.wav \
    --audio_encode_mode stream
