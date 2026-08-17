#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

CKPT_DIR="models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR="models/wav2vec2-base-960h"
COND_DIR="/home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/examples"

for i in $(seq 1 14); do

    id=$(printf "%03d" "$i")

    if [[ "$id" == "002" || "$id" == "009" || "$id" == "010" ]]; then
        AUDIO_PATH="${COND_DIR}/female_test.wav"
    else
        AUDIO_PATH="${COND_DIR}/male_test.wav"
    fi

    echo "========================================"
    echo "Processing: ${id}.png"
    echo "Audio: $(basename "$AUDIO_PATH")"
    echo "========================================"

    python generate_video.py \
        --ckpt_dir "$CKPT_DIR" \
        --wav2vec_dir "$WAV2VEC_DIR" \
        --model_type lite \
        --cond_image "${COND_DIR}/${id}.png" \
        --audio_path "$AUDIO_PATH" \
        --audio_encode_mode stream

done