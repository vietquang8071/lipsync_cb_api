#!/usr/bin/env bash

SOURCE="/home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/examples/output_pip_omnivoice.wav"
TARGET_DIR="/dev/shm/audio"

mkdir -p "$TARGET_DIR"

for i in $(seq 1 10); do
  cp "$SOURCE" "$TARGET_DIR/test${i}.wav"
done

ls -lh "$TARGET_DIR"/test*.wav
