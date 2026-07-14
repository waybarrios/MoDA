# LLaVA-MoRE + MoDA

This directory is a fork of [aimagelab/LLaVA-MORE](https://github.com/aimagelab/LLaVA-MORE) — a LLaVA-style visual instruction tuning codebase built around LLaMA-3.1/LLaMA-3 backbones — extended with **MoDA** (Modulation Adapter), the instruction-guided, channel-wise modulation module described in our paper *"MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs"* (ICML 2026, [arXiv:2506.01850](https://arxiv.org/abs/2506.01850)). See the [root README](../README.md) for the paper abstract, headline results across LLaVA-1.5 / LLaVA-MoRE / Qwen3-VL, and citation information. This directory contains everything needed to reproduce the LLaVA-MoRE + MoDA results specifically.

## Where is MoDA in the code?

| Concern | Location |
|---|---|
| MoDA module (adapter + builder) | `llava/model/multimodal_moda/builder.py` — `MoDAAdapter` (the cross-attention modulation block) and `build_moda()` (factory function) |
| Integration into the LLaVA forward pass | `llava/model/llava_arch.py` — `self.moda` is constructed on the model mixin, and `encode_images()` applies the modulation mask to the vision features before they reach the LLM |
| Training flag | `llava/train/train.py` — `--moda True` enables MoDA training (`--masking` is kept as a deprecated alias for backward compatibility with older configs/checkpoints) |

## Installation

1. Create and activate a fresh environment (Python 3.10 recommended, matching upstream LLaVA-MoRE):
   ```bash
   conda create -n llava-more-moda python=3.10 -y
   conda activate llava-more-moda
   ```
2. Install dependencies from this directory:
   ```bash
   cd llava-more
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Optional) Install [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [flash-attention](https://github.com/Dao-AILab/flash-attention) if not already pulled in by `requirements.txt` — both are used by the training scripts below.

This fork inherits its environment, data conventions, and most of its training/eval tooling from upstream [aimagelab/LLaVA-MORE](https://github.com/aimagelab/LLaVA-MORE); all credit for the base pipeline goes to that project. The Apache-2.0 license under which it was released is preserved in `LICENSE`.

## Pretrained checkpoints

Both LLaVA-MoRE 8B + MoDA variants from the paper are available on Hugging Face 🤗:

| Variant | Checkpoint |
|---|---|
| SigLIP-SO400M + S2 (paper main config) | [waybarrios/MoDA-LLaVA-MoRE-8B-SigLIP-S2](https://huggingface.co/waybarrios/MoDA-LLaVA-MoRE-8B-SigLIP-S2) |
| CLIP ViT-L/14-336 | [waybarrios/MoDA-LLaVA-MoRE-8B-CLIP](https://huggingface.co/waybarrios/MoDA-LLaVA-MoRE-8B-CLIP) |

Load them with this codebase:

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "waybarrios/MoDA-LLaVA-MoRE-8B-SigLIP-S2"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
)
```

Evaluation of all paper benchmarks uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Data

Training uses the standard LLaVA v1.5 instruction-tuning mix: a single JSON annotation file (665K examples, `llava_v1_5_mix665k.json`) plus the corresponding image folders (COCO, GQA, OCR-VQA, TextVQA, VisualGenome, etc.), following the same layout as upstream LLaVA / LLaVA-MoRE. Point `DATA_PATH` at the JSON file and `IMAGE_FOLDER` at the root directory containing the image subfolders — see the [official LLaVA data instructions](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning) for how to assemble this mix if you don't already have it.

## Training

Training follows the standard two-stage LLaVA protocol: stage 1 pretrains the vision-to-language projector with the LLM and vision encoder frozen; stage 2 introduces MoDA and fine-tunes it jointly with the LLM.

**Stage 1 — projector pretraining.** Use `scripts/more/07_pretrain_siglip.sh` as the base template for the SigLIP-S2 config used in the paper. As shipped, that script pretrains a SigLIP projector on a Vicuna-7B backbone; to reproduce the paper's LLaMA-3.1-8B + SigLIP-S2 setup, adapt it by switching the backbone/tokenizer to `--llm_backbone llama_3_1` (as in `scripts/more/11_pretrain_llama_31_acc_st_1.sh`) and adding the S2 flags (`--s2 True --s2_scales "384,768,1152"`, as in `scripts/more/05_pretrain_s2.sh`). For the CLIP-336 variant, `scripts/more/11_pretrain_llama_31_acc_st_1.sh` is the closest starting point (drop `--siglip`/`--s2`, set `--vision_tower openai/clip-vit-large-patch14-336`). All `scripts/more/*.sh` files have had their wandb/slurm machine-specific env lines stripped; fill in your own `cd`, `TOKENIZER_PATH`, and vision/LLM paths before running. The resulting `mm_projector.bin` (written under `--output_dir`) becomes `PRETRAIN_MM_ADAPTER` in stage 2. `scripts/extract_mm_projector.py` is an optional utility for pulling the projector weights out of a quantized checkpoint; it is not needed for a normal run.

**Stage 2 — MoDA fine-tuning.** Two canonical, ready-to-run scripts are provided:

- `scripts/finetune_moda_siglip_s2.sh` — the paper configuration: LLaMA-3.1-8B-Instruct + SigLIP-SO400M with S2 multi-scale features, `--moda True`.
- `scripts/finetune_moda_clip.sh` — the same recipe with a CLIP ViT-L/14-336 vision tower instead (no `--s2`/`--siglip` flags).

Both are configured entirely through environment variables (with sensible relative defaults) instead of hardcoded paths:

| Env var | Meaning | Default |
|---|---|---|
| `DATA_PATH` | Path to the LLaVA v1.5 mix JSON | `./playground/data/llava_v1_5_mix665k.json` |
| `IMAGE_FOLDER` | Root folder containing the training images | `./playground/data` |
| `PRETRAIN_MM_ADAPTER` | Stage-1 projector checkpoint (`mm_projector.bin`) | `./checkpoints/llava_more-siglip-s2-pretrain/mm_projector.bin` (or `-clip-pretrain` for the CLIP script) |
| `OUTPUT_DIR` | Where stage-2 checkpoints are written | `./checkpoints/llava_more-siglip-s2-moda` (or `-clip-moda`) |

Override any of them inline, e.g.:
```bash
DATA_PATH=/path/to/llava_v1_5_mix665k.json IMAGE_FOLDER=/path/to/images \
PRETRAIN_MM_ADAPTER=/path/to/mm_projector.bin \
bash scripts/finetune_moda_siglip_s2.sh
```
Both scripts also honor `TOKENIZER_PATH` and `MASTER_PORT` env vars, use `scripts/zero3.json` for DeepSpeed ZeRO-3, and write logs via `--report_to none` (set `--report_to wandb` yourself if desired).

## Evaluation

Evaluation scripts live under `llava/eval/`:

- `eval_science_qa.py` (and the GPT-4-assisted `eval_science_qa_gpt4*.py` variants) for ScienceQA.
- `eval_pope.py` for POPE hallucination evaluation.
- `model_vqa.py`, `model_vqa_loader.py`, `model_vqa_mmbench.py`, `model_vqa_science.py` for VQA-style benchmarks (GQA, MMBench, TextVQA, VizWiz, etc.).
- `run_cv.py` for CV-Bench; `run_llava.py` / `run_llava_images.py` for single-image / batch demo inference.

Several benchmarks require converting raw model outputs into the format expected by the official evaluation server or scorer; the corresponding converters live alongside the training scripts in `scripts/` (`convert_gqa_for_eval.py`, `convert_mmbench_for_submission.py`, `convert_mmvet_for_eval.py`, `convert_seed_for_submission.py`, `convert_sqa_to_llava.py` / `convert_sqa_to_llava_base_prompt.py`, `convert_vizwiz_for_submission.py`, `convert_vqav2_for_submission.py`). `run_cv.py --model-path` is a required argument (no default is shipped); `run_llava.py --model-path` defaults to `./checkpoints/llava_more-siglip-s2-moda`, matching the stage-2 SigLIP-S2 output directory above — point either at whatever checkpoint you trained.

## Legacy checkpoints

MoDA's parameters were originally named `masking.*` during development and were renamed to `moda.*` for this release. Checkpoints saved under the old naming still load correctly: the model's `load_state_dict` path remaps `masking.*` keys to `moda.*` automatically. If you'd rather have the checkpoint files themselves reflect the new naming (e.g. before uploading them somewhere), run `scripts/convert_legacy_moda_checkpoint.py /path/to/checkpoint_dir` — it rewrites `pytorch_model.bin` / sharded `.safetensors` files (and the corresponding shard index, if present) in place, replacing `masking.*` keys with `moda.*`.

## License / Acknowledgments

This directory is a fork of [aimagelab/LLaVA-MORE](https://github.com/aimagelab/LLaVA-MORE) and remains under the upstream Apache License 2.0 (see `LICENSE`). All credit for the base LLaVA-MoRE training/eval pipeline goes to its original authors. The MoDA-specific additions in this directory (`llava/model/multimodal_moda/`, the `--moda` flag, the canonical training scripts, and this README) are released under the MIT License that covers the rest of the [MoDA repository](../README.md).
