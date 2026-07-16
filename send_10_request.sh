#!/usr/bin/env bash

API_URL="http://localhost:8010/api/v1/lipsync/generate"
IMAGE_ID="001"

for i in $(seq 1 10); do
  AUDIO_PATH="/dev/shm/audio/test${i}.wav"

  curl -sS -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"image_id\": \"$IMAGE_ID\",
      \"audio_path\": \"$AUDIO_PATH\"
    }" \
    > "response_${i}.json" 2> "error_${i}.log" &
done

wait

echo "Đã gửi xong 10 request."