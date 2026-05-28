# MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs

Official repository for our ICML 2026 paper **"MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs."**

MoDA (**Mo**dulation **A**dapter) is a lightweight module that improves fine-grained visual grounding in Multimodal Large Language Models (MLLMs) through **instruction-guided, channel-wise modulation** of pre-aligned visual features. Unlike token-level methods such as Q-Former that perform additive feature selection, MoDA operates at the channel level via **multiplicative (Hadamard) modulation** on already-aligned features, giving fine-grained control over which embedding dimensions are relevant for each instruction — without modifying the underlying MLLM architecture and without additional supervision.

## Highlights

- **Channel-wise, not token-wise.** MoDA re-weights individual feature channels conditioned on the language instruction, addressing the *semantic entanglement* of visual patch representations.
- **Plug-and-play.** Operates post-alignment, at the adapter-to-LLM interface; complements rather than replaces existing adapters.
- **Minimal overhead.** Adds only **<1% FLOPs** and **3.7% parameters**.
- **No extra data or supervision.** Integrates into the standard two-stage LLaVA instruction-tuning pipeline.
- **Generalizes across architectures.** Consistent gains on **LLaVA-1.5**, **LLaVA-MoRE**, and **Qwen3-VL** (including non-CLIP encoders).

## Method

Given pre-aligned visual features `V_aligned` from the MLLM adapter and the language token embeddings `T`, MoDA learns a modulation function `F(·)` implemented as a stack of cross-attention Transformer layers (visual features as the target sequence, instruction embeddings as memory). A linear projection followed by a sigmoid produces a soft channel-wise mask in `[0, 1]`, which is applied element-wise:

```
Ṽ_aligned = V_aligned ⊙ σ(W · F(T, V_aligned))
```

The modulated features are then passed to the LLM for autoregressive decoding. Training follows the two-stage LLaVA protocol: (1) train the visual adapter with the vision encoder and LLM frozen; (2) introduce MoDA (Xavier-initialized) and fine-tune MoDA and the LLM jointly under a standard cross-entropy objective.

## Results

MoDA delivers consistent improvements across 12 benchmarks and three MLLM families. Selected highlights:

| Family | Benchmark | Baseline → MoDA |
|---|---|---|
| LLaVA-1.5 (Vicuna-7B) | MMVP | 24.0 → **36.0** (+12.0) |
| LLaVA-MoRE (SigLIP-S2) | ScienceQA | 77.1 → **81.9** (+4.8) |
| LLaVA-MoRE (SigLIP-S2) | POPE | 86.0 → **87.7** |
| Qwen3-VL-2B-Instruct | ScienceQA | 79.3 → **84.2** (+4.9) |
| Qwen3-VL-2B-Instruct | RealWorldQA | 64.7 → **68.8** (+4.1) |
| Qwen3-VL-2B-Instruct | GQA | 59.4 → **63.2** (+3.8) |

Gains are strongest on fine-grained, vision-centric, and hallucination tasks, and scale with the quality of the visual encoder — confirming that the improvements stem from architectural design rather than parameter scaling.

## Benchmarks

Evaluation spans three categories:

- **Visual Question Answering:** GQA, ScienceQA, MMBench, RealWorldQA, ChartQA
- **Vision-Centric:** LLaVA-Wild, MM-Vet, MMStar, V\*Bench, CV-Bench
- **Hallucination Detection:** POPE, MMVP

## Getting Started

The codebase builds on the official [LLaVA](https://github.com/haotian-liu/LLaVA) / LLaVA-MoRE pipelines. Full training and evaluation code, configuration files, and pretrained checkpoints will be released in this repository.

```bash
git clone https://github.com/waybarrios/MoDA.git
cd MoDA
# environment setup and usage instructions coming soon
```

## Citation

If you find MoDA useful in your research, please cite:

```bibtex
@inproceedings{barrios2026moda,
  title     = {MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs},
  author    = {Barrios, Wayner and Villa, Andr\'es and Leon Alcazar, Juan C. and Jin, SouYoung and Ghanem, Bernard},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by startup funds provided by Dartmouth College. In addition, the research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST) - Center of Excellence for Generative AI, under award number 5940.

## License

Released under the MIT License.
