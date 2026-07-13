# Qwen3-VL + MoDA

This directory implements **MoDA** (Modulation Adapter) — the instruction-guided, channel-wise modulation module described in our paper *"MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs"* (ICML 2026, [arXiv:2506.01850](https://arxiv.org/abs/2506.01850)) — for [Qwen3-VL](https://huggingface.co/Qwen). See the [root README](../README.md) for the paper abstract, headline results across LLaVA-1.5 / LLaVA-MoRE / Qwen3-VL, and citation information. This directory contains everything needed to reproduce the Qwen3-VL-2B + MoDA results specifically; the code also supports Qwen3-VL-8B training.

Unlike the LLaVA-1.5 and LLaVA-MoRE forks, this is not a fork of an existing training repo: it's a small custom `transformers`-compatible model definition (loaded via `trust_remote_code=True`) plus training configs for [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) and an evaluation plugin for [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Where is MoDA in the code?

| Concern | Location |
|---|---|
| MoDA module | `model/moda_adapter.py` — `MoDADecoderLayer` (self-attn → cross-attn → FFN, built from `nn.Linear` only, no `nn.MultiheadAttention`/`nn.TransformerDecoderLayer` so it stays DeepSpeed ZeRO-3 compatible) and `MoDAAdapter` (stacks the decoder layers, produces the sigmoid mask, and exposes `get_sparsity_loss()`) |
| Integration into the Qwen3-VL forward pass | `model/modeling_qwen3_vl_moda.py` — `Qwen3VLModelWithMoDA._apply_moda()` is called from `forward()` right after `get_image_features()` (the vision encoder + merger) and before `inputs_embeds.masked_scatter(image_mask, image_embeds)` (the point where visual tokens are written into the text sequence). `Qwen3VLForConditionalGenerationWithMoDA.forward()` adds `moda_adapter.get_sparsity_loss()` to the LM loss. |
| Config / auto-registration | `model/modeling_qwen3_vl_moda.py` — `Qwen3VLMoDAConfig` extends `Qwen3VLConfig` with `moda_*` fields, and the module registers itself with `AutoConfig`/`AutoModel`/`AutoModelForCausalLM` at import time so `trust_remote_code=True` picks it up automatically. |
| Model directory assembly | `scripts/prepare_model_dir.py` — symlinks the base Qwen3-VL weights and writes a `config.json` with `auto_map` pointing at the classes above (see Installation → Step 5 and Training below). |

## Architecture

```
Original Qwen3-VL:     Vision Encoder → Merger → [insert into text] → LLM
With MoDA:              Vision Encoder → Merger → MoDA Adapter → [insert into text] → LLM
                                                      ↑
                                                 text context
                                              (cross-attention)
```

The MoDA adapter generates a **soft modulation mask** over visual tokens using cross-attention with language embeddings:

```
ẽ_V = V ⊙ σ(W · F(T, V))
```

where `F` is a transformer decoder (self-attention + cross-attention + FFN), `W` is a linear projection, and `σ` is sigmoid.

**Key design choices:**
- Uses `nn.Linear` for all projections (DeepSpeed ZeRO-3 compatible, no `nn.MultiheadAttention`)
- Sparsity regularization loss (L1 on sigmoid mask values) added to the main LM loss
- Frozen vision tower + frozen projector; only the language model + MoDA adapter are trained (see `freeze_vision_tower` / `freeze_multi_modal_projector` / `freeze_language_model` in the training configs)
- MoDA is injected **after** the vision encoder + merger and **before** visual tokens are scattered into the text sequence

### MoDA Adapter Hyperparameters

The adapter scales based on the base model's hidden size (see `MODEL_CONFIGS` in `scripts/prepare_model_dir.py`):

| Parameter          | 2B     | 8B     | Description |
|-------------------|--------|--------|-------------|
| Embedding dim     | 2048   | 4096   | Matches base model hidden size (automatic) |
| Hidden dim (FFN)  | 512    | 1024   | Feed-forward intermediate dimension |
| Attention heads   | 16     | 32     | Multi-head attention heads |
| Decoder layers    | 1      | 1      | Number of transformer decoder layers |
| Dropout           | 0.1    | 0.1    | Dropout rate in attention and FFN |
| Sparsity lambda   | 1e-3   | 1e-3   | L1 regularization weight on mask |
| **Extra params**  | **~33M** | **~100M** | Parameters added by MoDA adapter |

## Results (2B)

Trained on **LLaVA v1.5 mix665k** (1 epoch, 4x GPUs). Evaluated against the base Qwen3-VL-2B-Instruct model:

| Benchmark         | Metric            | Base    | MoDA    | Delta    | Δ%     |
|------------------|-------------------|---------|---------|----------|--------|
| GQA              | exact_match       | 0.5937  | 0.6378  | +0.0441  | +7.4%  |
| ScienceQA        | exact_match       | 0.7930  | 0.8208  | +0.0278  | +3.5%  |
| RealWorldQA      | exact_match       | 0.6471  | 0.6876  | +0.0405  | +6.3%  |
| ChartQA          | relaxed_overall   | 0.8000  | 0.7480  | -0.0520  | -6.5%  |
| MMStar           | average           | 0.5385  | 0.5456  | +0.0071  | +1.3%  |
| POPE             | pope_accuracy     | 0.8941  | 0.8956  | +0.0014  | +0.2%  |
| MMVet            | gpt_eval_score    | 51.93   | 53.35   | +1.42    | +2.7%  |

**Average (0-1 scale benchmarks): +1.6%** improvement over the base model (over 6 benchmarks, excluding MMVet which uses a 0-100 scale).

## Project Structure

```
qwen3-vl/
├── model/                                      # Core MoDA implementation
│   ├── __init__.py                             # Module exports
│   ├── moda_adapter.py                         # MoDADecoderLayer + MoDAAdapter
│   └── modeling_qwen3_vl_moda.py               # Qwen3-VL + MoDA integration
├── configs/
│   ├── qwen3vl_moda_full_sft.yaml              # 2B training config (DDP)
│   ├── qwen3vl_8b_moda_full_sft_z2.yaml        # 8B training config (ZeRO-2)
│   ├── qwen3vl_8b_moda_full_sft_z3.yaml        # 8B training config (ZeRO-3)
│   ├── ds_z2_config.json                       # DeepSpeed ZeRO-2 config
│   └── ds_z3_config.json                       # DeepSpeed ZeRO-3 config
├── scripts/
│   ├── prepare_model_dir.py                    # Assemble local model dir (2B/8B)
│   ├── train.sh                                # Train 2B (DDP)
│   ├── train_8b.sh                             # Train 8B (DeepSpeed ZeRO-2/3)
│   ├── eval_base.sh                            # Eval base 2B
│   ├── eval_moda.sh                            # Eval MoDA 2B
│   ├── eval_all.sh                             # Parallel eval 2B
│   ├── eval_base_8b.sh                         # Eval base 8B
│   ├── eval_moda_8b.sh                         # Eval MoDA 8B
│   ├── eval_all_8b.sh                          # Parallel eval 8B
│   └── compare_results.py                      # Compare base vs MoDA results
├── data/
│   ├── convert_llava_to_llamafactory.py        # Dataset format converter
│   └── dataset_info.json                       # LlamaFactory dataset metadata
├── lmms_eval_integration/
│   └── qwen3_vl_moda.py                        # lmms-eval model wrapper for MoDA checkpoints
└── requirements.txt
```

`configs/` also ships a few extra DeepSpeed variants (`ds_z1_config.json`, `ds_z2_no_overlap_config.json`, `ds_z2_safe_config.json`) and exploratory LlamaFactory YAMLs (`qwen3vl_4b_moda_v2*.yaml`, `qwen3vl_moda_v2.yaml`) from earlier experimentation. They are not referenced by `train.sh` / `train_8b.sh` and are not required to reproduce the results above; the canonical configs are the three listed in the tree.

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU(s) with CUDA 12.8+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) (or any Python 3.11 environment manager)

| Model | Min VRAM (training) | Recommended setup |
|-------|-------------------|-------------------|
| 2B    | ~24 GB per GPU    | 1-4x GPUs, DDP (no DeepSpeed needed) |
| 8B    | ~40 GB per GPU (ZeRO-3) / ~80 GB per GPU (ZeRO-2) | 4-8x A100 80GB with DeepSpeed |

**A correctly pinned Python environment matters a lot here** — `transformers` support for Qwen3-VL, the `auto_map`/`trust_remote_code` loading path, and DeepSpeed ZeRO-3 compatibility are all version-sensitive. Use the versions in `requirements.txt` as your starting point rather than "whatever is already installed."

### Step 1: Create a Python environment

```bash
conda create -n qwen3vl_moda python=3.11 -y
conda activate qwen3vl_moda
```

### Step 2: Install PyTorch

Install PyTorch matching your CUDA version. For CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions, see https://pytorch.org/get-started/locally/

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 3: Install Python dependencies

```bash
cd qwen3-vl
pip install -r requirements.txt
```

This installs `transformers>=4.57.0`, `accelerate`, `datasets`, `llamafactory`, `deepspeed`, `qwen-vl-utils`, `peft`, `trl`, etc. — see `requirements.txt` for the full pinned list.

### Step 4: Install lmms-eval (evaluation framework)

Evaluation uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), a unified evaluation framework for multimodal models. This repo does not bundle lmms-eval; install it and register the `qwen3_vl_moda` wrapper against it:

1. Install lmms-eval into your environment, either from PyPI or from source:
   ```bash
   pip install lmms-eval
   # or, from a clone of the official repo:
   # git clone https://github.com/EvolvingLMMs-Lab/lmms-eval && cd lmms-eval && pip install -e .
   ```
   This has been verified against lmms-eval 0.7.1, which already ships a base `qwen3_vl` model (`lmms_eval/models/simple/qwen3_vl.py`) — the MoDA wrapper subclasses it, so it must be present.
2. Copy the MoDA wrapper into lmms-eval's `models/simple/` directory:
   ```bash
   cp lmms_eval_integration/qwen3_vl_moda.py "$(python -c 'import lmms_eval, os; print(os.path.dirname(lmms_eval.__file__))')/models/simple/qwen3_vl_moda.py"
   ```
3. Register the model in lmms-eval's model registry. Open `lmms_eval/models/__init__.py` in your lmms-eval install and add an entry to the `AVAILABLE_SIMPLE_MODELS` dict (it maps a model id to a class name; lmms-eval dynamically imports `lmms_eval.models.simple.<model_id>.<ClassName>` from it):
   ```python
   AVAILABLE_SIMPLE_MODELS = {
       ...
       "qwen3_vl_moda": "Qwen3_VL_MoDA",
       ...
   }
   ```
   `qwen3_vl_moda.py` already self-registers with lmms-eval's `@register_model("qwen3_vl_moda")` decorator (`lmms_eval/api/registry.py`), but that registry is only populated once the module has actually been imported — adding the `AVAILABLE_SIMPLE_MODELS` entry above is what makes `--model qwen3_vl_moda` resolve to it and trigger that import.
4. Verify:
   ```bash
   python -m lmms_eval --model qwen3_vl_moda --tasks gqa --limit 1 --model_args pretrained=Qwen/Qwen3-VL-2B-Instruct
   ```

### Step 5: Download the base model

The base models are downloaded automatically from Hugging Face on first use. The download location follows the standard `HF_HOME` environment variable (defaults to `~/.cache/huggingface` if unset):

```bash
export HF_HOME="/path/to/your/hf_cache"   # optional — omit to use ~/.cache/huggingface
```

To pre-download:

```bash
# 2B model
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
"

# 8B model
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
"
```

### Step 6: Set PYTHONPATH

`train.sh` / `train_8b.sh` set `PYTHONPATH` to the project root automatically. If you run anything manually (e.g. a one-off `prepare_model_dir.py` invocation, or the lmms-eval commands above from outside a script), export it yourself so `model/` can be imported:

```bash
export PYTHONPATH="/path/to/qwen3-vl:${PYTHONPATH}"
```

---

## Training

### Overview

Training uses [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) for full supervised fine-tuning (SFT). The workflow is:

1. Convert the dataset from LLaVA format to LlamaFactory ShareGPT format (`data/convert_llava_to_llamafactory.py`)
2. Assemble a local model directory that combines the base Qwen3-VL weights with the MoDA config extensions and custom modeling code (`scripts/prepare_model_dir.py`)
3. Run training with `llamafactory-cli train` via `scripts/train.sh` (2B) or `scripts/train_8b.sh` (8B)

`scripts/train.sh` and `scripts/train_8b.sh` run steps 1 and 2 automatically the first time (skipped on subsequent runs if their outputs already exist), so a manual run is only needed if you want to customize the conversion or model-dir arguments.

### Step 1: Prepare training data

You need the LLaVA v1.5 mix665k JSON (`llava_v1_5_mix665k.json`, 665K conversations) and the corresponding images, following the standard [LLaVA instruction-tuning data layout](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning).

Convert it to LlamaFactory format:

```bash
python data/convert_llava_to_llamafactory.py \
    --input /path/to/llava_v1_5_mix665k.json \
    --output data/llava_v1_5_mix665k_llamafactory.json \
    --image_base_dir /path/to/images
```

**Arguments:**
- `--input`: Path to the original LLaVA JSON (665k entries with `conversations` + `image` fields)
- `--output`: Where to write the converted LlamaFactory JSON
- `--image_base_dir`: Base directory prepended to each image path (the LLaVA JSON has relative paths like `coco/train2017/000033471.jpg`)

The converter maps `{"from": "human", ...}` → `{"role": "user", ...}` and extracts `images` into a separate field.

`train.sh` / `train_8b.sh` run this automatically with `--input "${LLAVA_JSON:-data/llava_v1_5_mix665k.json}"` and `--image_base_dir "${IMAGE_BASE_DIR:-data/images}"` if `data/llava_v1_5_mix665k_llamafactory.json` doesn't already exist — override `LLAVA_JSON` / `IMAGE_BASE_DIR` to point at your own paths.

The dataset metadata is defined in `data/dataset_info.json` under the key `llava_mix665k`, which is what `dataset: llava_mix665k` in the training configs refers to.

### Step 2: Prepare the model directory

```bash
# For 2B model
python scripts/prepare_model_dir.py --model-size 2b

# For 8B model
python scripts/prepare_model_dir.py --model-size 8b
```

The script finds the base model snapshot under `$HF_HOME/hub/models--Qwen--Qwen3-VL-*/snapshots/` (or `$HF_HOME/models--Qwen--Qwen3-VL-*/snapshots/`), then:
1. Copies `config.json` and adds `auto_map` (so `trust_remote_code=True` loads the MoDA classes) + MoDA hyperparameters scaled for the chosen model size
2. Symlinks all weight files (`.safetensors`, tokenizer, etc.) from the original model — no weights are duplicated on disk
3. Copies `model/moda_adapter.py` and `model/modeling_qwen3_vl_moda.py` into the new model directory

This produces `qwen3_vl_2b_moda/` or `qwen3_vl_8b_moda/` in the project root, which is what `model_name_or_path` in the training configs points to.

`train.sh` / `train_8b.sh` also run this automatically (with the defaults above) if the expected `config.json` doesn't already exist.

### Training: 2B model

```bash
# Single GPU (DDP)
bash scripts/train.sh 1

# Multi-GPU (4 GPUs with DDP — used for the published results)
bash scripts/train.sh 4
```

Uses `configs/qwen3vl_moda_full_sft.yaml` (no DeepSpeed — DDP fits the 2B model in single-GPU memory).

| Parameter | Value |
|-----------|-------|
| DeepSpeed | None (DDP) |
| Batch size per GPU | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 4 x 8 x N_GPUs |
| Learning rate | 2e-5 |
| Epochs | 1 |

Checkpoints are written to `checkpoints/qwen3vl_moda_full_sft/` (relative to the project root, per `output_dir` in the config).

### Training: 8B model

The 8B model requires **DeepSpeed** for distributed training. Two configurations are provided:

#### Option A: ZeRO-2 (optimizer + gradient sharding)

Each GPU holds a full copy of the model weights. **Requirements:** 4+ GPUs with ~80 GB VRAM each (e.g., A100 80GB, H100 80GB).

```bash
bash scripts/train_8b.sh 4 2   # 4 GPUs, ZeRO-2
bash scripts/train_8b.sh 8 2   # 8 GPUs, ZeRO-2
```

Config: `configs/qwen3vl_8b_moda_full_sft_z2.yaml` (DeepSpeed config: `configs/ds_z2_config.json`)

| Parameter | Value |
|-----------|-------|
| Batch size per GPU | 1 |
| Gradient accumulation | 32 |
| Learning rate | 1e-5 |
| Epochs | 1 |

#### Option B: ZeRO-3 (full parameter sharding)

Shards weights + optimizer + gradients across GPUs — less VRAM per GPU, more communication overhead. **Requirements:** 4+ GPUs with ~40 GB VRAM each (e.g., A100 40GB, A6000 48GB).

```bash
bash scripts/train_8b.sh 4 3   # 4 GPUs, ZeRO-3
bash scripts/train_8b.sh 8 3   # 8 GPUs, ZeRO-3
```

Config: `configs/qwen3vl_8b_moda_full_sft_z3.yaml` (DeepSpeed config: `configs/ds_z3_config.json`)

| Parameter | Value |
|-----------|-------|
| Batch size per GPU | 2 |
| Gradient accumulation | 16 |
| Learning rate | 1e-5 |
| Epochs | 1 |

Checkpoints are written to `checkpoints/qwen3vl_8b_moda_full_sft_z2/` (or `_z3/`). The final checkpoint directory contains the model weights, a `config.json` with `auto_map` pointing at the MoDA classes, `moda_adapter.py` + `modeling_qwen3_vl_moda.py` (auto-saved by LlamaFactory alongside the weights), and the tokenizer/training state.

#### Before running 8B training

```bash
export CONDA_ENV="/path/to/your/conda/envs/qwen3vl_moda"   # optional — activates that env's bin/ on PATH; omit to use whatever env is already active
export HF_HOME="/path/to/your/hf_cache"                      # optional — defaults to ~/.cache/huggingface
```

All config YAML files use paths relative to the project root — run the scripts from `qwen3-vl/`.

---

## Evaluation

### Overview

Evaluation uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (see Installation → Step 4 for how to register the MoDA wrapper). Two model types are used:
- `qwen3_vl` — loads a base Qwen3-VL model (native to lmms-eval; works for both 2B and 8B)
- `qwen3_vl_moda` — loads a MoDA checkpoint with `trust_remote_code=True` via `lmms_eval_integration/qwen3_vl_moda.py` (works for both 2B and 8B)

### Benchmarks

| Benchmark     | Task ID        | Metric              | Scale | Description |
|--------------|----------------|---------------------|-------|-------------|
| GQA          | `gqa`          | `exact_match`       | 0-1   | Visual reasoning over scene graphs |
| ScienceQA    | `scienceqa`    | `exact_match`       | 0-1   | Science knowledge + visual reasoning |
| RealWorldQA  | `realworldqa`  | `exact_match`       | 0-1   | Real-world visual question answering |
| ChartQA      | `chartqa`      | `relaxed_overall`   | 0-1   | Chart and plot understanding |
| MMStar       | `mmstar`       | `average`           | 0-1   | Comprehensive multimodal benchmark |
| POPE         | `pope`         | `pope_accuracy`     | 0-1   | Object hallucination detection |
| MMVet        | `mmvet`        | `gpt_eval_score`    | 0-100 | GPT-4-evaluated open-ended VQA (requires an OpenAI API key) |

### Evaluation: 2B model

```bash
# Base model, all benchmarks
bash scripts/eval_base.sh 0        # arg = GPU id

# MoDA model, all benchmarks (defaults to checkpoints/qwen3vl_moda_full_sft)
bash scripts/eval_moda.sh 0

# Both in parallel (base on GPU 4, MoDA on GPU 5 by default)
bash scripts/eval_all.sh
```

Under the hood, each script loops over `gqa scienceqa realworldqa chartqa mmstar` and, if `OPENAI_API_KEY` is set, also runs `mmvet`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
    --model qwen3_vl_moda \
    --model_args pretrained=checkpoints/qwen3vl_moda_full_sft \
    --tasks gqa \
    --batch_size 1 \
    --output_path eval_results/moda
```

### Evaluation: 8B model

```bash
# Base 8B
bash scripts/eval_base_8b.sh 0

# MoDA 8B — defaults to the ZeRO-2 checkpoint; pass a path to evaluate the ZeRO-3 one instead
bash scripts/eval_moda_8b.sh 0
bash scripts/eval_moda_8b.sh 0 /path/to/checkpoints/qwen3vl_8b_moda_full_sft_z3

# Both in parallel
bash scripts/eval_all_8b.sh
```

### Run a single benchmark manually

```bash
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
    --model qwen3_vl_moda \
    --model_args pretrained=/path/to/checkpoint \
    --tasks chartqa \
    --batch_size 1 \
    --output_path eval_results/moda
```

Add `--limit N` to any command to run a quick sanity check on `N` samples per benchmark instead of the full set.

### MMVet (GPT-based evaluation)

MMVet requires an OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-key-here"
bash scripts/eval_moda.sh 0       # 2B
bash scripts/eval_moda_8b.sh 1    # 8B
```

If `OPENAI_API_KEY` is not set, the eval scripts automatically skip MMVet and only run the 5 non-GPT benchmarks.

### Compare results

```bash
python scripts/compare_results.py --base eval_results/base --moda eval_results/moda        # 2B
python scripts/compare_results.py --base eval_results/base_8b --moda eval_results/moda_8b  # 8B
```

This scans each `output_path` for `*_results.json` files and prints a base-vs-MoDA delta table (like the Results table above), averaging only over 0-1 scale benchmarks.

---

## Troubleshooting

**`ValueError: Unrecognized configuration class Qwen3VLMoDAConfig for AutoModelForCausalLM`** — this means the assembled model directory (`qwen3_vl_2b_moda/` or `qwen3_vl_8b_moda/`) has a stale/partial `config.json` or stale copies of `moda_adapter.py` / `modeling_qwen3_vl_moda.py` (e.g. left over from an older run, or `trust_remote_code` wasn't set when loading). Fix:
1. Make sure your environment is activated and up to date (`pip install -r requirements.txt`).
2. Delete and regenerate the model directory:
   ```bash
   rm -rf qwen3_vl_2b_moda/
   python scripts/prepare_model_dir.py --model-size 2b
   ```
3. Double-check the training config (`configs/qwen3vl_moda_full_sft.yaml` or the 8B equivalents) has `model_name_or_path` pointing at that same directory and `trust_remote_code: true` set.

**"Model not found in HF cache"** — the base model hasn't been downloaded, or `HF_HOME` doesn't point at the cache you expect. Set `HF_HOME` explicitly and re-run the download snippet in Installation → Step 5. `scripts/prepare_model_dir.py` checks both `$HF_HOME/hub/models--...` and `$HF_HOME/models--...` layouts.

**`auto_map` / `trust_remote_code` reminders** — `config.json` in the assembled model directory must contain the `auto_map` block written by `prepare_model_dir.py` (mapping `AutoConfig`/`AutoModel`/`AutoModelForCausalLM`/`AutoModelForImageTextToText` to the classes in `modeling_qwen3_vl_moda.py`), and any code loading the model (training config, eval wrapper, or a manual `from_pretrained` call) must pass `trust_remote_code=True`. Without it, `transformers` falls back to the stock Qwen3-VL classes and silently drops MoDA.

**DeepSpeed ZeRO-3 + MoDA** — the adapter is implemented entirely with `nn.Linear`/`nn.LayerNorm` (see `model/moda_adapter.py`) specifically so ZeRO-3 parameter partitioning works correctly; if you modify it, avoid `nn.MultiheadAttention` or `nn.TransformerDecoderLayer`, which are not ZeRO-3 safe.

---

## Acknowledgments

- Base model: [Qwen3-VL](https://huggingface.co/Qwen) (2B and 8B) by Alibaba
- MoDA concept adapted from [LLaVA-MORE](https://github.com/aimagelab/LLaVA-MORE)
- Training framework: [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- Evaluation framework: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- Training data: LLaVA v1.5 mix665k (see the [official LLaVA data instructions](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning))

This directory's MoDA-specific code (`model/`, `scripts/`, `configs/`, `data/`, `lmms_eval_integration/`, this README) is released under the MIT License that covers the rest of the [MoDA repository](../README.md).
