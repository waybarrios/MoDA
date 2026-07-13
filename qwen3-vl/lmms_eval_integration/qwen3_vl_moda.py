"""Qwen3-VL + MoDA model wrapper.

Loads the model with ``AutoModelForImageTextToText`` and ``trust_remote_code=True``
so that the custom MoDA adapter weights are picked up from the checkpoint directory.
Everything else (processor, generation, etc.) is identical to the base Qwen3_VL class.
"""

import re
from typing import Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("qwen3_vl_moda")
class Qwen3_VL_MoDA(Qwen3_VL):
    """Qwen3_VL variant that loads a MoDA-adapted checkpoint via trust_remote_code."""

    def __init__(
        self,
        pretrained: str = "checkpoints/qwen3vl_moda_full_sft",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Skip Qwen3_VL.__init__ — call the grandparent directly
        lmms.__init__(self)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # --- KEY DIFFERENCE: use AutoModelForImageTextToText + trust_remote_code ---
        eval_logger.info(f"Loading MoDA model from {pretrained} with trust_remote_code=True")
        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()

        # Verify MoDA adapter is present
        has_moda = any("moda" in name.lower() for name, _ in self._model.named_modules())
        if has_moda:
            eval_logger.info("MoDA adapter modules detected in loaded model")
        else:
            eval_logger.warning("WARNING: No MoDA adapter modules found — weights may not have loaded correctly!")

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
