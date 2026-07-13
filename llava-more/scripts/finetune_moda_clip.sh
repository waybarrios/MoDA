#!/bin/bash
# Stage-2 finetuning: LLaVA-MoRE (LLaMA-3.1-8B, CLIP ViT-L/14-336) + MoDA.
# Set the env vars below to your local data paths.

export PYTHONPATH="."
export OMP_NUM_THREADS=1
export TOKENIZER_PATH=${TOKENIZER_PATH:-aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning}
MASTER_PORT=${MASTER_PORT:-29500}

EPOCHS=1
LLM_PATH="meta-llama/Llama-3.1-8B-Instruct"
VISION_TOWER="openai/clip-vit-large-patch14-336"
DATA_PATH=${DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data}
PRETRAIN_MM_ADAPTER=${PRETRAIN_MM_ADAPTER:-./checkpoints/llava_more-clip-pretrain/mm_projector.bin}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/llava_more-clip-moda}

deepspeed --master_port $MASTER_PORT llava/train/train_mem.py \
  --deepspeed "./scripts/zero3.json" \
  --model_name_or_path "$LLM_PATH" \
  --llm_backbone "llama_3_1" \
  --llm_pad_token "pad" \
  --version "llama_3_1" \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --vision_tower "$VISION_TOWER" \
  --pretrain_mm_mlp_adapter "$PRETRAIN_MM_ADAPTER" \
  --mm_projector_type "mlp2x_gelu" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio "pad" \
  --moda True \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --report_to none \
  --run_name "llava_more_clip_moda"
