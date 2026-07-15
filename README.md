# lipsync_cb_api
###  🔧 Installation
#### 1. Create a Conda environment
```bash
conda create -n flashhead python=3.10
conda activate flashhead
```
#### 2. Install PyTorch on CUDA
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
#### 3. Install other dependencies
```bash
pip install -r requirements.txt
```
#### 4. FlashAttention installation:
```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

-- If it takes a long time, we recommend the way below.
1. download wheel file from [here](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)
2. pip install xxx.whl

#### 5. SageAttention installation (Optional)
```bash
pip install sageattention==2.2.0 --no-build-isolation
```

#### 6. FFmpeg installation
```bash
# Ubuntu / Debian
apt-get install ffmpeg
# CentOS / RHEL
yum install ffmpeg ffmpeg-devel
```
or
```bash
# Conda (no root required) 
conda install -c conda-forge ffmpeg==7
```
### 🤗 Model download
Only download lite version
| Model Component | Description | Link |
| :--- | :--- | :---: |
| `SoulX-FlashHead-1_3B` | Our 1.3B model| 🤗 [Huggingface](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | wav2vec2-base-960h | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h) |

After download, put the weight into app/thirdparty/SoulX_FlashHead/models/ like this:

app/thirdparty/SoulX_FlashHead/models/
├── SoulX-FlashHead-1_3B/
│   ├── Model_Lite/
│   └── VAE_LTX/
└── wav2vec2-base-960h/
    ├── config.json
    └── model.safetensors

```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h


