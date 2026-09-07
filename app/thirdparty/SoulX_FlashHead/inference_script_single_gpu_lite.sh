CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite \
    --cond_image /home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/avatar/001.png \
    --audio_path /home/cbrain/ai_digital_hero/LipSync_CB/app/thirdparty/SoulX_FlashHead/examples/female_test.mp3 \
    --audio_encode_mode stream