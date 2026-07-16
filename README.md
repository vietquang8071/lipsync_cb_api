# lipsync_cb_api
###  🔧 Installation
#### 0. Git clone this repo

#### 1. Create a Conda environment
```bash
conda create -n flashhead python=3.10
conda activate flashhead
```
#### 2. Install PyTorch on CUDA
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
If your NVIDIA driver only supports CUDA version lower than 12.8, using this:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

To check CUDA, using this command:
 ```bash
 nvidia-smi
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

```bash
app/thirdparty/SoulX_FlashHead/models/
├── SoulX-FlashHead-1_3B/
│   ├── Model_Lite/
│   └── VAE_LTX/
└── wav2vec2-base-960h/
    ├── config.json
    └── model.safetensors
```

```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

### For docker using (Make sure that you have docker on your machine/server):

#### 1) Following above steps

#### 2) Create a folder in /dev/shm/
```bash
sudo mkdir /dev/shm/audio
```
Also make sure that you have these folders in lipsync_cb_api:
```bash
/storage
/images
/logs
```
#### Then, grant read and write permissions to this directory.
```bash
sudo chmod 777 /dev/shm/audio
```

#### 3) 
```bash
cd path/to/lipsync_cb_api
```

#### 4) Run build docker
```bash
docker compose build
docker compose up
```

#### 5) To check container status
```bash
docker compose ps
```