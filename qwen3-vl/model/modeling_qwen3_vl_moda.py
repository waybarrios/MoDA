"""
Qwen3-VL + MoDA: Custom model that injects the MoDA adapter between
the vision encoder output and the language model input.

This file is designed to be loaded by LlamaFactory with trust_remote_code=True.

===========================================================================
WHERE MoDA IS INJECTED (in Qwen3VLModelWithMoDA.forward):
===========================================================================

  Original Qwen3-VL forward flow:
  --------------------------------
  1. inputs_embeds = embed_tokens(input_ids)           # text embeddings
  2. image_embeds, deepstack = get_image_features(...)  # vision encoder
  3. image_embeds = torch.cat(image_embeds, dim=0)      # flatten images
  4. inputs_embeds.masked_scatter(image_mask, image_embeds)  # insert into sequence

  With MoDA:
  --------------------------------
  1. inputs_embeds = embed_tokens(input_ids)           # text embeddings
  2. image_embeds, deepstack = get_image_features(...)  # vision encoder
  >>>  image_embeds = MoDA(image_embeds, text_context)  # <-- MoDA HERE
  3. image_embeds = torch.cat(image_embeds, dim=0)      # flatten images
  4. inputs_embeds.masked_scatter(image_mask, image_embeds)  # insert into sequence

  The MoDA adapter uses cross-attention where:
    - Query (tgt):    visual tokens  (N_img, 2048)
    - Memory (kv):    text tokens    (N_text, 2048)
    - Output:         modulated visual tokens * sigmoid(mask)
===========================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)

# is_torchdynamo_compiling was removed from transformers >=5.0;
# fall back to the standard PyTorch API.
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import is_torchdynamo_compiling
except ImportError:
    from torch.compiler import is_compiling as is_torchdynamo_compiling
from transformers.cache_utils import Cache
from transformers.utils import auto_docstring
from transformers.utils.generic import TransformersKwargs
from typing import Unpack

from .moda_adapter import MoDAAdapter


class Qwen3VLMoDAConfig(Qwen3VLConfig):
    """Extended config that adds MoDA adapter parameters."""
    model_type = "qwen3_vl"  # Keep same model_type for LlamaFactory compatibility

    def __init__(
        self,
        moda_hidden_dim: int = 512,
        moda_num_heads: int = 16,
        moda_num_layers: int = 1,
        moda_dropout: float = 0.1,
        moda_sparsity_lambda: float = 1e-3,
        moda_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moda_hidden_dim = moda_hidden_dim
        self.moda_num_heads = moda_num_heads
        self.moda_num_layers = moda_num_layers
        self.moda_dropout = moda_dropout
        self.moda_sparsity_lambda = moda_sparsity_lambda
        self.moda_enabled = moda_enabled


class Qwen3VLModelWithMoDA(Qwen3VLModel):
    """
    Qwen3VLModel with MoDA adapter injected between vision encoding and
    language model input.

    The MoDA adapter sits AFTER the vision encoder + merger (projector)
    and BEFORE the visual tokens are scattered into the text sequence.
    """
    config_class = Qwen3VLMoDAConfig

    def __init__(self, config: Qwen3VLMoDAConfig):
        super().__init__(config)

        # Build MoDA adapter using the LLM's hidden_size
        embedding_dim = config.text_config.hidden_size  # 2048 for Qwen3-VL-2B
        self.moda_adapter = MoDAAdapter(
            embedding_dim=embedding_dim,
            hidden_dim=getattr(config, "moda_hidden_dim", 512),
            num_heads=getattr(config, "moda_num_heads", 16),
            num_layers=getattr(config, "moda_num_layers", 1),
            dropout=getattr(config, "moda_dropout", 0.1),
            sparsity_lambda=getattr(config, "moda_sparsity_lambda", 1e-3),
        )
        self.moda_enabled = getattr(config, "moda_enabled", True)

    def _apply_moda(
        self,
        image_embeds_list,
        inputs_embeds: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Apply MoDA cross-attention modulation to image embeddings.

        Args:
            image_embeds_list: tuple/list of (N_i, D) tensors, one per image
            inputs_embeds: (B, seq_len, D) full input embeddings (includes placeholders)
            input_ids: (B, seq_len) token ids, used to identify text vs image tokens

        Returns:
            Modulated flat image embeddings (total_image_tokens, D)
        """
        if not self.moda_enabled or len(image_embeds_list) == 0:
            return torch.cat(image_embeds_list, dim=0)

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        # Extract text-only embeddings as language context for cross-attention
        if input_ids is not None:
            text_mask = (input_ids != image_token_id) & (input_ids != video_token_id)
            lang_embeds_list = []
            for b in range(inputs_embeds.shape[0]):
                lang_embeds_list.append(inputs_embeds[b, text_mask[b], :])

            max_text_len = max(e.shape[0] for e in lang_embeds_list)
            lang_padded = torch.zeros(
                len(lang_embeds_list), max_text_len, inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
            for i, e in enumerate(lang_embeds_list):
                lang_padded[i, :e.shape[0], :] = e
        else:
            lang_padded = inputs_embeds

        # Apply MoDA to each image's tokens with text context
        modulated_parts = []
        for img_embed in image_embeds_list:
            # img_embed: (N_i, D) -> (1, N_i, D) for batch processing
            img_3d = img_embed.unsqueeze(0)
            lang_ctx = lang_padded[:1]  # (1, text_len, D)
            modulated = self.moda_adapter(img_3d, lang_ctx)  # (1, N_i, D)
            modulated_parts.append(modulated.squeeze(0))  # (N_i, D)

        return torch.cat(modulated_parts, dim=0)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Qwen3VLModelOutputWithPast]:
        """Forward with MoDA adapter injected for image features."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            # get_image_features returns (list_of_embeds, deepstack_embeds)
            image_embeds_list, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )

            # ============================================================
            # >>> MoDA INJECTION POINT <<<
            # Modulate visual tokens with language context BEFORE
            # scattering them into the text sequence.
            # ============================================================
            image_embeds = self._apply_moda(image_embeds_list, inputs_embeds, input_ids)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds_list, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # --- Aggregate visual masks and deepstack embeds (unchanged) ---
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # --- Position IDs (unchanged from original) ---
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # --- Language model forward (unchanged) ---
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGenerationWithMoDA(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL + MoDA for conditional generation.
    Drops in as a replacement for Qwen3VLForConditionalGeneration.
    Adds MoDA sparsity loss to the language modeling loss.
    """
    config_class = Qwen3VLMoDAConfig

    def __init__(self, config: Qwen3VLMoDAConfig):
        # Call grandparent init to avoid double model creation
        super(Qwen3VLForConditionalGeneration, self).__init__(config)

        # Replace self.model with MoDA-enabled version
        self.model = Qwen3VLModelWithMoDA(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        """Forward pass with MoDA sparsity loss added to the main loss."""

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )
            # Ensure MoDA params always have gradients for DeepSpeed ZeRO
            # gradient sync (prevents NCCL deadlock on text-only batches)
            moda_used = self.model.moda_adapter.last_mask is not None

            # Add MoDA sparsity regularization to the main loss
            moda_loss = self.model.moda_adapter.get_sparsity_loss()
            if isinstance(moda_loss, torch.Tensor) and moda_loss.item() > 0:
                loss = loss + moda_loss

            if not moda_used:
                loss = loss + 0.0 * sum(p.sum() for p in self.model.moda_adapter.parameters())

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )


# ---------------------------------------------------------------------------
# Auto-register with HuggingFace Auto classes so that
# AutoModelForCausalLM.from_pretrained() works even when trust_remote_code
# is only passed to AutoConfig (which triggers this module's import).
# ---------------------------------------------------------------------------
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("qwen3_vl_moda", Qwen3VLMoDAConfig)
    AutoModel.register(Qwen3VLMoDAConfig, Qwen3VLModelWithMoDA)
    AutoModelForCausalLM.register(
        Qwen3VLMoDAConfig, Qwen3VLForConditionalGenerationWithMoDA
    )
except Exception:
    pass
