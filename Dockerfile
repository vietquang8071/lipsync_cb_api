ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG XFORMERS_VERSION=0.0.28.post3
ARG PYTORCH_CUDA_INDEX=cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LIPSYNC_STORAGE_DIR=/opt/lipsync/storage/lipsync \
    LIPSYNC_LOG_DIR=/opt/lipsync/logs \
    SOULX_CKPT_DIR=/opt/lipsync/app/thirdparty/SoulX_FlashHead/models/SoulX-FlashHead-1_3B \
    SOULX_WAV2VEC_DIR=/opt/lipsync/app/thirdparty/SoulX_FlashHead/models/wav2vec2-base-960h \
    SOULX_CONFIG_PATH=/opt/lipsync/app/thirdparty/SoulX_FlashHead/flash_head/configs/infer_params.yaml

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        ninja-build \
        python3-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lipsync

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}" \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "xformers==${XFORMERS_VERSION}"

COPY requirements.txt /tmp/requirements.txt

# python-multipart is required by FastAPI's UploadFile endpoints. PyYAML and
# einops are imported directly by the vendored SoulX inference code.
RUN python3 -m pip install \
        --requirement /tmp/requirements.txt \
        python-multipart \
        PyYAML \
        einops \
    && rm /tmp/requirements.txt

COPY app /opt/lipsync/app

RUN mkdir -p \
        /opt/lipsync/logs \
        /opt/lipsync/storage/lipsync

VOLUME ["/opt/lipsync/storage", "/opt/lipsync/logs"]

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3)" || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
