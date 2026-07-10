import sys
import copy
import yaml
import torch
import librosa
import threading
import numpy as np

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from app.utils import save_video


PROJECT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_DIR / "app"
SOULX_DIR = APP_DIR / "thirdparty" / "SoulX_FlashHead"

if not SOULX_DIR.exists():
    raise FileNotFoundError(f"SoulX directory not found: {SOULX_DIR}")

if str(SOULX_DIR) not in sys.path:
    sys.path.insert(0, str(SOULX_DIR))


import copy
import yaml
import torch
from loguru import logger
# from app.thirdparty.SoulX_FlashHead.flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
# from app.thirdparty.SoulX_FlashHead.flash_head.src.distributed.usp_device import get_device, get_parallel_degree


class SoulXModel():
  def __init__(self,
               cpkt_dir:str,
               wav2vec_dir:str,
               model_type: str="lite",
               config_path: str | None = None,
               world_size: int=1
               ):
    self.cpkt_dir = cpkt_dir
    self.wav2vec_dir = wav2vec_dir
    self.model_type = model_type
    self.config_path = config_path or str(SOULX_DIR / "flash_head/configs/infer_params.yaml")
    self.world_size = world_size

    self.pipeline = None
    self.infer_params = None
    self.device = None
    self.loaded = False
    self._lock = threading.RLock()

  def _check_loaded(self):
    if not self.loaded or self.pipeline is None:
      raise RuntimeError("SoulXModel is not loaded. Call model.load() first")
  
  def load(self):
    """
    Load mô hình lên GPU. Gọi 1 lần
    """
    with self._lock:
      if self.loaded:
        logger.info("SoulXModel already loaded.")
        return self

      # Lazy import tránh lỗi khởi tạo CUDA và các tham số trong process cha 
      from app.thirdparty.SoulX_FlashHead.flash_head.src.pipeline.flash_head_pipeline import (FlashHeadPipeline,)
      from app.thirdparty.SoulX_FlashHead.flash_head.src.distributed.usp_device import (get_device, get_parallel_degree,)

      with open(self.config_path, "r") as f:
        self.infer_params=yaml.safe_load(f)
      
      ulysses_degree, ring_degree = get_parallel_degree(
        self.world_size,
        self.infer_params["num_heads"]
      )

      self.device = get_device(ulysses_degree, ring_degree)

      logger.info(
        f"Loading SoulX-FlashHead on device={self.device}, "
        f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
      )

      self.pipeline = FlashHeadPipeline(
        checkpoint_dir=self.cpkt_dir,
        model_type=self.model_type,
        wav2vec_dir=self.wav2vec_dir,
        device=self.device,
        use_usp=(self.world_size > 1),
      )


      motion_frames_latent_num = self.infer_params["motion_frames_latent_num"]
      motion_frames_num = (motion_frames_latent_num - 1) * self.pipeline.config.vae_stride[0] + 1
      self.infer_params["motion_frames_num"] = motion_frames_num

      if self.model_type=="pretrained":
        self.infer_params["sample_steps"]=20
      else:
        self.infer_params["sample_steps"]=4
      
      self.loaded=True
      logger.info("SoulX-FlashHead loaded successfully")

      return self
  
  def _prepare_base_data(
      self,
      cond_image_path_or_dir:str,
      base_seed: int = 42,
      use_face_crop: bool = True,
  ):
    self._check_loaded()

    self.pipeline.prepare_params(
      cond_image_path_or_dir=cond_image_path_or_dir,
      target_size=(
        self.infer_params["height"],
        self.infer_params["width"]
      ),
      frame_num=self.infer_params["frame_num"],
      motion_frames_num=self.infer_params["motion_frames_num"],
      sampling_steps=self.infer_params["sample_steps"],
      seed=base_seed,
      shift=self.infer_params["sample_shift"],
      color_correction_strength=self.infer_params["color_correction_strength"],
      use_face_crop=use_face_crop
      )
    
  def _get_audio_embedding(
      self,
      audio_array,
      audio_start_idx: int=-1,
      audio_end_idx: int=-1
  ):
    """
    Embedding audio
    """
    self._check_loaded()
    audio_embedding=self.pipeline.preprocess_audio(
      audio_array,
      sr=self.infer_params["sample_rate"],
      fps=self.infer_params["tgt_fps"],
    )

    if audio_start_idx==-1 or audio_end_idx==-1:
      audio_start_idx=0
      audio_end_idx = audio_embedding.shape[0]

    indicies = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indicies = (
      torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indicies.unsqueeze(0)
    )

    center_indicies = torch.clamp(
      center_indicies,
      min=0,
      max=audio_end_idx - 1
    )

    audio_embedding = audio_embedding[center_indicies][None, ...].contiguous()
    return audio_embedding
  
  @torch.inference_mode()
  def infer(self, audio_embedding):
    """
    Generate video frames
    """
    self._check_loaded()

    audio_embedding = audio_embedding.to(self.pipeline.device)

    sample = self.pipeline.generate(audio_embedding)
    sample_frames = (
        (((sample + 1) / 2)
          .permute(1, 2, 3, 0)
          .clip(0, 1) * 255)
        .contiguous()
    )

    return sample_frames
  
  def _get_infer_params(self):
    self._check_loaded()
    return copy.deepcopy(self.infer_params)
  
  def generate_video(self,
      image_path:Path,
      audio_path:Path,
      output_path:Path,
      seed:int,
      use_face_crop:bool
  ) -> None:
    with self._lock:
      if not self.loaded:
        self.load()

      self._prepare_base_data(
        cond_image_path_or_dir=str(image_path),
        base_seed=seed,
        use_face_crop=use_face_crop,
      )

      infer_params = self._get_infer_params()
      sample_rate = infer_params["sample_rate"]
      tgt_fps = infer_params["tgt_fps"]
      cached_audio_duration = infer_params["cached_audio_duration"]
      frame_num = infer_params["frame_num"]
      motion_frames_num = infer_params["motion_frames_num"]

      slice_len = frame_num - motion_frames_num
      audio_slice_len = slice_len * sample_rate // tgt_fps

      speech_array, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True
      )

      remainder = len(speech_array) % audio_slice_len
      if remainder > 0:
        pad_length = audio_slice_len - remainder
        speech_array = np.concatenate(
          [speech_array, np.zeros(pad_length, dtype=speech_array.dtype)]
        )

      cached_audio_length = int(sample_rate * cached_audio_duration)
      audio_end_idx = int(cached_audio_duration * tgt_fps)
      audio_start_idx = audio_end_idx - frame_num

      audio_dq = deque([0.0] * cached_audio_length, maxlen=cached_audio_length)
      audio_chunks = speech_array.reshape(-1, audio_slice_len)

      generated_frames = []
      
      for chunk_idx, speech_chunk in enumerate(audio_chunks):
        logger.info(f"Generating lipsync chunk {chunk_idx}")

        audio_dq.extend(speech_chunk.tolist())
        audio_array = np.array(audio_dq)

        audio_embedding = self._get_audio_embedding(
          audio_array=audio_array,
          audio_start_idx=audio_start_idx,
          audio_end_idx=audio_end_idx
        )

        video = self.infer(audio_embedding=audio_embedding)
        video = video[motion_frames_num:]

        generated_frames.append(video.cpu())

        if not generated_frames:
          raise RuntimeError("No video frames were generated.")
        
      save_video(
        frames_list=generated_frames,
        video_path=output_path,
        audio_path=audio_path,
        fps=tgt_fps
      )


  def unload(self):
    """
    Giải phóng model khỏi GPU
    """
    with self._lock:
      if self.pipeline is not None:
        del self.pipeline
        self.pipeline = None

      self.loaded = False

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

      logger.info("SoulX unloaded.")
