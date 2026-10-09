# Open-weight LLMs for inference on up to 8× NVIDIA B300

Date checked: 2026-10-09. Research and planning only. No serving profile was added.

Question: which self-hostable, open-weight models available on this date can run inference on at most eight NVIDIA B300 GPUs, and which of those are plausible alternatives to DeepSeek-V3.2 if the goal is quality closer to Claude 4.5.

This repository has no prior research-note convention. The local-LLM plan lives in `scripts/local_llm/PLAN.md`. This note is the research companion, not a change to that plan.

## How to read the numbers

**Confirmed** means the figure is copied from an NVIDIA document, a vendor model card, a Hugging Face model API (`safetensors` dtype counts and `usedStorage`), or a serving recipe published by the model vendor or by vLLM.

**Estimate** means a byte-budget calculated here from those figures. Estimates assume all experts stay resident on GPU. Expert offload to CPU (KTransformers and similar) is a different serving design and is not budgeted below.

Hugging Face `usedStorage` is converted with \(10^9\) bytes = 1 GB. GPU capacities use NVIDIA’s labeled GB. \(8 \times 288 = 2304\) GB. NVIDIA rounds that node total to 2.3 TB.

A shared score on SWE-bench Verified is not evidence that a local model matches Claude 4.5. Scaffolds, thinking budgets, sample counts, and which Claude generation was scored all change the number. Later open-model cards usually compare themselves with Claude Opus 4.6, Opus 4.8, or Claude Fable 5, which are different models from the Claude 4.5 family.

## Hardware

| Fact | Source |
| --- | --- |
| DGX B300: 8× NVIDIA B300 Blackwell Ultra, GPU memory \(8 \times 288\) GB = 2.3 TB | [DGX B300 User Guide](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html) |
| HGX B300 SXM: 288 GB HBM3e per GPU, 2.30 TB per node, configuration “up to 2304 GB”, bandwidth up to 8 TB/s per GPU, GPU-to-GPU NVLink 1800 GB/s | [HGX AI Factory components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html) |
| Blackwell Ultra maximum HBM is 288 GB HBM3e at up to 8 TB/s. SM count and HBM capacity vary by SKU. The 192 GB figure in that table is Blackwell (B200-class), not B300. | [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) |

This note assumes the 288 GB SXM SKU documented for DGX B300 and HGX B300. A lower B300 SKU has less memory than the budgets below.

Serving reserves are conventions, not hardware specs. Official GLM-5 and many vLLM recipes set `--gpu-memory-utilization` around 0.85–0.95, leaving the rest for activations and the KV cache. At 0.90, eight 288 GB GPUs expose about 2074 GB. One sequence at the context lengths already used in this repo (32,768 or 131,072 tokens) is a much smaller KV consumer than a 1M-token, high-concurrency server.

Mixture-of-experts weight memory follows **total** parameters, not activated parameters. Activated parameters change arithmetic and speed.

## Baseline: DeepSeek-V3.2 and Claude 4.5

### DeepSeek-V3.2 (what the local profile serves today)

Confirmed from the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) and the Hugging Face model API (snapshot last modified 2025-12-01):

- 685B parameters. The card’s base-model table on the V4 page lists the V3.2 base as 671B total / 37B activated; the instruct checkpoint’s extra parameters are the next-token prediction block (`num_nextn_predict_layers: 1` in `config.json`).
- Checkpoint dtype: FP8 e4m3 block quantization, 128×128 scales (`quant_method: fp8`). Safetensors split: 681.4B FP8, 3.95B BF16, small FP32. `usedStorage` 689.5 GB.
- Context in config: `max_position_embeddings` 163,840, with YaRN from 4,096. The local profile caps requests at 131,072.
- License: MIT. Tool calling is part of the V3.2 chat format. The Speciale variant is the same architecture and does not support tool calling.
- Local serving in this repo is planned as tensor parallel 4. Four B300s are \(4 \times 288 = 1152\) GB. Weights at 689 GB leave roughly 460 GB before the utilization reserve. That fits, with room for a 128k sequence.

**Estimate, latent KV only.** Config stores a compressed KV of `kv_lora_rank` 512 plus `qk_rope_head_dim` 64, across 61 layers. In BF16 that is \((512+64) \times 2 \times 61 = 70{,}272\) bytes per token, about 9.2 GB for one 131,072-token sequence. The sparse-attention indexer is extra and is not sized in the card. Treat 9 GB as a floor, not the full runtime KV.

### Claude 4.5 (closed; the quality reference, not a local candidate)

Anthropic’s own posts, retrieved 2026-10-09:

- [Claude Sonnet 4.5](https://www.anthropic.com/news/claude-sonnet-4-5) (September 2025): SWE-bench Verified **77.2%**, simple scaffold (bash and string-replace edit), average of 10 trials, no test-time compute, 200K thinking budget, full 500-problem set. OSWorld **61.4%** on OSWorld-Verified, 100 max steps, average of 4 runs.
- [Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5) (24 November 2025): at the highest effort, Opus 4.5 exceeds Sonnet 4.5 by **4.3 percentage points** on SWE-bench Verified; at medium effort it matches Sonnet 4.5’s best score on that benchmark and uses 76% fewer output tokens. The launch post’s methodology says evals used a 64K thinking budget, 200K context, default effort high, five trials, with exceptions for SWE-bench Verified (no thinking budget) and Terminal-Bench (128K thinking budget). The post text does not print an absolute Opus SWE-bench percentage.
- Later vendor cards repeatedly cite **80.9** for Opus 4.5 on SWE-bench Verified (Kimi K2.5, GLM-5, Qwen3.5). That 80.9 is their citation, not a number copied from the Anthropic post above. Those same cards disagree with each other on other Opus 4.5 cells (HLE is the clearest example). Use 80.9 only as “the figure those labs attribute to Opus 4.5,” and keep it tied to the card that printed it.

Haiku 4.5 exists ([announcement](https://www.anthropic.com/news/claude-haiku-4-5), SWE-bench Verified 73.3% on Anthropic’s scaffold). It is the small Claude 4.5, so it is the wrong target if the goal is the top of that generation.

## What is worth testing

Ordered for a first local trial on one 8× B300 node. “GPUs” means 288 GB B300s with weights resident and one in-flight 128k-class request, unless a vendor recipe says otherwise.

### 1. DeepSeek-V4-Pro — test this first

[deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). MIT. Preview card dated 2026; repo last modified 2026-06-22.

Confirmed:

- 1.6T total, 49B activated, 1M context. Instruct checkpoint precision: **FP4 experts, FP8 for most other parameters** (`expert_dtype: fp4`, `quant_method: fp8`).
- Safetensors: 1.573T INT8 elements (packed FP4), 23.2B FP8, 2.82B BF16. Total counted parameters 1.599T. `usedStorage` **864.8 GB**.
- At 1M context, DeepSeek says V4-Pro uses 27% of V3.2’s single-token FLOPs and **10% of V3.2’s KV cache**. That ratio is a vendor claim, not a measured byte count on B300.
- Think Max mode: they recommend a context window of at least 384K. Sampling `temperature = 1.0`, `top_p = 1.0`.
- Their Max-mode table (their harness, compared with **Opus 4.6 Max**, not Claude 4.5): SWE-bench Verified 80.6 vs Opus 4.6 Max 80.8; Terminal-Bench 2.0 67.9 vs 65.4; GPQA Diamond 90.1 vs 91.3; HLE 37.7 vs 40.0. Non-think SWE-bench Verified is 73.6, in the same band as the V3.2 numbers other labs publish (~73). The gain they advertise is the high/max thinking modes.

Fit: 865 GB of files on 2304 GB leaves about 1440 GB raw. Four GPUs (1152 GB) still clear the file size. At 0.90 utilization, four GPUs expose about 1037 GB, about 170 GB above the file size. That is enough for a 128k request if the engine keeps experts packed and the 10% KV claim is even roughly right. Eight GPUs are the comfortable setting for Think Max at 384k. If a server upcasts experts to FP8, weight memory moves toward ~1.6 TB and four GPUs are no longer enough; eight still are (2304 − 1600 ≈ 700 GB raw).

This is the direct successor of the checkpoint the repo already plans to serve, under the same MIT license.

### 2. GLM-5.3-BF16 — strongest coding claim that is still one node, if the license is acceptable

[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16). Created 2026-08-25. Hugging Face tags the license `other`. GLM-5.2, which Z.ai says is the same base model, is MIT. Read `LICENSE` in the 5.3 repo before any use that is not a private experiment.

Confirmed:

- Same architecture family as GLM-5 (`GlmMoeDsaForCausalLM`). Safetensors: **753.3B BF16**. `usedStorage` **1506.7 GB**. No quantization block in the config. The card says every gain over GLM-5.2 is post-training, and it advertises a 1M-token context on the 5.2 card that this checkpoint continues.
- Z.ai’s own table (Claude Code harness for several coding rows; judge and timeout details are in the card footnotes), compared with **Opus 4.8**, not Claude 4.5: Terminal-Bench 2.1 88.2 vs Opus 4.8 85.0 and vs Kimi K3 88.3; DeepSWE v1.1 66.9 vs Opus 4.8 58.0 and vs Kimi K3 67.5; HLE with tools 62.5 vs Opus 4.8 57.9. Terminal-Bench 3.0 is 28.3 vs Opus 4.8 21.1 and vs Fable 5 33.7. NL2Repo is 58.0 vs Opus 4.8 69.7. These are Z.ai’s measurements. Harnesses are not Anthropic’s SWE-bench scaffold.

Fit: 1507 GB of BF16 weights. Six GPUs are 1728 GB raw and only about 1555 GB at 0.90 utilization, which does not comfortably hold 1507 GB plus KV. Seven GPUs are 2016 GB raw and about 1814 GB at 0.90, leaving ~300 GB after weights. Eight GPUs leave ~800 GB raw. A 128k single request fits on seven GPUs under the latent-KV estimate; a 1M context should be tried on all eight. There is no official FP8 copy of this checkpoint in the repo that was inspected. Community quantizations exist and were not treated as vendor quality.

GLM-5, GLM-5.1, and GLM-5.2 are the same ~1.51 TB BF16 shape. For a first GLM trial, 5.3 is the current flagship. GLM-5.2 remains the MIT option at the same byte cost, with lower scores on Z.ai’s own 5.3 table.

GLM-5’s earlier card is the one that still names **Claude Opus 4.5** directly: SWE-bench Verified 77.8 vs their cited Opus 4.5 80.9 vs DeepSeek-V3.2 73.1; Terminal-Bench 2.0 (Terminus 2) 56.2 vs 59.3 vs 39.3; GPQA-Diamond 86.0 vs 87.0 vs 82.4. That is the cleanest “closer to Claude 4.5 than V3.2, still short on SWE-bench Verified” statement in this set, and it is about GLM-5, not 5.3. 5.3’s card does not repeat the Opus 4.5 column.

### 3. Kimi-K3 — highest upside, tightest node, newest serving stack

[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3). Kimi K3 License (not MIT). Repo created 2026-06-13, last modified 2026-09-02.

Confirmed:

- 2.8T total, 104B activated, 1,048,576 context, native text and image. Quantization-aware **MXFP4 weights / MXFP8 activations**. Routed experts are MXFP4; attention, shared experts, router, `lm_head`, and vision stay higher precision (the ignore list in `quantization_config`).
- Safetensors: 2.723T UINT8 packed elements, 57.2B BF16. Counted total 2.780T. `usedStorage` **1561.0 GB**.
- [vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3): hardware minimum **8× GB300** (Blackwell Ultra superchip; same 288 GB-class GPU as B300). Listed footprints: MXFP4 **1680 GB** (recipe text still says “estimated”), FP8 **3360 GB**. The published command uses tensor parallel 8, `--gpu-memory-utilization 0.95`, `--max-model-len 1048576`, `--kv-cache-dtype fp8`. The recipe also says multi-node is for real production traffic, and it pins a K3-specific vLLM image.

Fit: the 1561 GB checkpoint fits in 2304 GB with about 740 GB raw left. The recipe’s 1680 GB MXFP4 estimate also fits, with less room: at the recipe’s 0.95 utilization, eight GPUs expose about 2189 GB, leaving about 500 GB if the 1680 GB figure is the live weight footprint. The FP8 footprint of 3360 GB does **not** fit on eight B300s. Serve the MXFP4 checkpoint. Do not plan to park Boltz on the same eight GPUs; the recipe’s minimum is the whole node, at a 1M context. A shorter context frees KV, and that tradeoff is an estimate until measured.

The card’s benchmark table compares K3 (max effort) with Claude Fable 5, Claude Opus 4.8, and GPT-5.6, not with Claude 4.5. Examples, all with the harness caveats in the card: Terminal-Bench 2.1 88.3 vs Opus 4.8 84.6 vs Fable 5 88.0; GPQA Diamond 93.5 vs Opus 4.8 91.0; HLE-Full 43.5 / 56.0 (no tools / with tools) vs Opus 4.8 49.8 / 57.9. Several coding numbers use Kimi Code for K3 and a different harness for Claude. Moonshot’s footnotes document that. Being near Opus 4.8 on those rows is evidence about a **later** Claude, on Moonshot’s suite. It is a reason to test K3. It is not a Claude 4.5 match.

### 4. Kimi-K2.6 — same quality band on SWE-bench Verified, much easier to place

[moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6). Modified MIT. Same shape as K2.5: 1T total, 32B activated, 256K context, MLA, native INT4 on routed experts. `usedStorage` **595.3 GB**. Safetensors: 1.015T INT32 packed elements plus 12.2B BF16.

The vLLM recipe for K2.5, which K2.6 says to reuse, lists a minimum of **714 GB** for the packed INT4 checkpoint and an example of 8× H200. Three B300s are 864 GB. Four are 1152 GB, which covers 714 GB of weights and leaves hundreds of GB for KV. This is the model that can keep a Boltz GPU on the same node without squeezing the KV cache.

K2.6’s card compares with **Opus 4.6 (max)**, in-house SWE harness, 10-run average: SWE-bench Verified **80.2** vs 80.8; SWE-bench Pro 58.6 vs 53.4; Terminal-Bench 2.0 (Terminus-2) 66.7 vs 65.4; GPQA-Diamond 90.5 vs 91.3. The predecessor K2.5 card is the one that still names **Claude 4.5 Opus**: SWE-bench Verified 76.8 vs 80.9 vs DeepSeek-V3.2 73.1; Terminal-Bench 2.0 50.8 vs 59.3 vs 46.4; GPQA-Diamond 87.6 vs 87.0 vs 82.4. K2.6’s 80.2 sits next to the 80.9 those labs attribute to Opus 4.5, on a different scaffold and against a newer Claude column. That is “worth a trial,” not equivalence.

### 5. Qwen3.5-397B-A17B — Apache-2.0, and the card actually has a Claude 4.5 column

[Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B). Apache-2.0. 397B total, 17B activated, native context 262,144, extensible to 1,010,000. BF16 safetensors, `usedStorage` **806.8 GB**. The card says to keep at least 128K context so thinking still works.

Four B300s (1152 GB) hold 807 GB of weights with about 345 GB raw left. Eight are unnecessary for a single 128k–256k stream.

Their language table names **Claude 4.5 Opus**: SWE-bench Verified 76.4 vs 80.9; SWE-bench Multilingual 69.3 vs 77.5; Terminal-Bench 2 52.5 vs 59.3; GPQA 88.4 vs 87.0; HLE 28.7 vs 30.8; MMLU-Pro 87.8 vs 89.5. Footnotes change TAU2-Bench (airline fixes from the Opus 4.5 system card) and BrowseComp (two context-folding strategies, 69.0 and 78.6). On this card, Qwen3.5 is ahead of the DeepSeek-V3.2 numbers published elsewhere on GPQA and near them on coding, and it remains behind the Opus 4.5 column on SWE-bench Verified and Terminal-Bench. Seventeen billion active parameters should be cheaper per token than V3.2’s 37B or V4-Pro’s 49B. That is an inference from the active-parameter counts, not a measured tokens/s figure.

Qwen3.6 open weights found on this pass are [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Qwen3.6-Plus, Qwen3.7-Max, and Qwen3.8-Max appear as **columns in other labs’ tables**. No public weight repo for those Max/Plus names was verified here. They are not self-hosting candidates on the evidence in this note.

### 6. DeepSeek-V4-Flash — fits beside Boltz, trails Pro on the hard agent rows

[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash). MIT. 284B total, 13B activated, 1M context, same FP4-expert + FP8 mix. `usedStorage` **159.6 GB**. One B300 (288 GB) holds the files with about 128 GB raw left, which is tight once the runtime allocator takes its reserve. Two GPUs are the practical choice.

DeepSeek’s own mode table: Flash Max SWE-bench Verified 79.0 and Terminal-Bench 2.0 56.9, versus Pro Max 80.6 and 67.9. They say Flash Max can approach Pro on reasoning with a larger thinking budget and trails on knowledge and the hardest agent workflows. Useful as a cheap control. It is not the model most likely to move past V3.2 toward Claude 4.5 on agentic coding.

## Memory summary

| Checkpoint | Precision (confirmed) | On-disk size | 8× B300 (2304 GB) | Practical GPU count |
| --- | --- | --- | --- | --- |
| DeepSeek-V3.2 | FP8 e4m3 | 689 GB | Fits, large KV margin | 4, as already planned |
| DeepSeek-V4-Pro | FP4 experts + FP8 rest | 865 GB | Fits | 4 if packed FP4 and context stays near 128k; 8 for Think Max at 384k |
| DeepSeek-V4-Flash | FP4 experts + FP8 rest | 160 GB | Fits | 2 |
| Kimi-K2.6 | INT4 routed experts, BF16 elsewhere | 595 GB | Fits | 3–4 (vLLM minimum 714 GB on the K2.5 recipe) |
| Qwen3.5-397B-A17B | BF16 | 807 GB | Fits | 4 |
| GLM-5.3-BF16 | BF16 | 1507 GB | Fits | 7–8. Six is too tight at 0.90 utilization |
| Kimi-K3 | MXFP4 weights, MXFP8 activations | 1561 GB | Fits the MXFP4 checkpoint | 8, per the vLLM GB300 minimum. FP8 footprint 3360 GB does not fit |
| Nemotron 3 Ultra NVFP4 | NVFP4 | 705 GB | Fits | NVIDIA lists a minimum of 4× B300 |
| MiniMax-M2.7 | FP8 | 230 GB | Fits | 2 |

“Practical GPU count” is an estimate from file size, the 0.90 utilization convention, and vendor minimums. It is not a measured `nvidia-smi` result on this cluster.

## Rejected or not a first test

| Candidate | Why it is not in the test order |
| --- | --- |
| DeepSeek-V3.2-Speciale | Same 685B FP8 shape. The card says it is deep-reasoning only and **does not support tool calling**. V4-Pro supersedes it for a quality trial. |
| GLM-5, GLM-5.1, GLM-5.2 | Same ~1.51 TB BF16 footprint as GLM-5.3. 5.2 is the MIT fallback if 5.3’s license is a problem. 5 and 5.1 are older post-trains of that base. |
| Kimi-K2.5 | Same 595 GB INT4 shape as K2.6. K2.6 is the newer post-train. K2.5’s card remains the one with an explicit Claude 4.5 Opus column. |
| NVIDIA Nemotron 3 Ultra 550B-A55B | [NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4): 550B total / 55B active, up to 1M context, OpenMDW-1.1, minimum **4× B300** (also 4× B200 / GB200 / GB300 or 8× H100). `usedStorage` 704.7 GB. Their Nemo Evaluator numbers: SWE-bench Verified **70.7 BF16 / 69.5 NVFP4**, Terminal-Bench 2.1 56.4 / 53.9, GPQA 87.0 / 87.9, HLE without tools 26.7 / 26.1. On their own SWE-bench Verified column this sits with or below DeepSeek-V3.2’s published 70–73 band, and behind K2.6 / V4-Pro / GLM self-reports. Fits easily. Not the closest path to Claude 4.5. |
| MiniMax-M2.7 | [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7). FP8, 228.7B parameters, `usedStorage` 230.1 GB, so it fits on two B300s. NVIDIA’s Nemotron card describes it as 230B-A10B and scores SWE-bench Verified at 75.3 in NVIDIA’s harness. The [license](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE) allows non-commercial use and requires prior written authorization for commercial use. Ten billion active parameters and that license put it behind the five models above for this goal. |
| Llama 4 Maverick | [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct). Released 5 April 2025. 400B total, 17B active, 1M context, BF16 in the example loader (about 800 GB if loaded in BF16; that byte figure is an estimate from the parameter count). Llama 4 Community License. Knowledge cutoff August 2024. It does not appear in the 2026 frontier tables cited above. Memory is fine. The quality case for beating V3.2 toward Claude 4.5 is not there. |
| Llama 4 Scout | 109B total / 17B active, 10M context. Too small, and a 10M cache is a KV problem even when weights fit. |
| gpt-oss-120b / gpt-oss-20b | OpenAI’s August 2025 open models. They fit. None of the 2026 flagship cards treat them as Claude 4.5 peers. |
| Qwen3-32B-FP8, Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B | Fit on one or two B300s. The 32B FP8 checkpoint is already the local smoke-test profile. These are not Claude 4.5-class alternatives. |
| Qwen3.6-Plus, Qwen3.7-Max, Qwen3.8-Max | Names in other labs’ comparison tables. Downloadable weights were not verified. |
| Kimi-K3 as FP8 or BF16 | vLLM lists an FP8 footprint of 3360 GB. BF16 at 2 bytes × 2.78T parameters is about 5.6 TB. Both exceed 2304 GB. Only the MXFP4 checkpoint is in range. |
| Any dense or MoE checkpoint stored as BF16 above ~2.0 TB | Eight 288 GB GPUs cannot hold the weights and a KV cache. Examples: a 1.6T model left in BF16 (~3.2 TB), or Kimi-K3 upcast off MXFP4. |

## Benchmark caveats that apply to every “closer to Claude” sentence

1. Claude 4.5 means Sonnet 4.5, Haiku 4.5, and Opus 4.5, announced in 2025. Open-weight cards from mid and late 2026 mostly score **later** Claude models. A win against Opus 4.6 or 4.8 is not a measurement against Opus 4.5.
2. Anthropic’s published Opus 4.5 SWE-bench statement is a **+4.3 point** gap over Sonnet 4.5’s 77.2 on Anthropic’s scaffold. The absolute 80.9 that Kimi, Z.ai, and Qwen print for Opus 4.5 is their number.
3. SWE-bench Verified scores in this note come from different scaffolds: Anthropic’s bash/edit scaffold, Qwen’s and Kimi’s in-house tool sets, Z.ai’s OpenHands or Claude Code runs, NVIDIA’s Nemo Evaluator. Gaps of a few points are inside harness noise.
4. Thinking mode dominates. DeepSeek-V4-Pro non-think SWE-bench Verified is 73.6; Max is 80.6. A local server that disables thinking will not show the card’s headline.
5. No independent lab reran this whole set on one scaffold for this note. There is no single ranking.

## Recommendation

Serve **DeepSeek-V4-Pro** first, on four to eight B300s, MIT, as the successor of the checkpoint already planned. Keep experts in FP4.

If that trial still sits short of the Claude 4.5 behavior you care about, the next two are **GLM-5.3-BF16** (whole node, confirm the non-MIT license) and **Kimi-K2.6** (four GPUs, Modified MIT, SWE-bench Verified in the band other labs assign to Opus 4.5). **Kimi-K3** is the quality ceiling that still fits, and only as MXFP4 on all eight GPUs with the vLLM K3 recipe. **Qwen3.5-397B-A17B** is the Apache-2.0 option and, on Qwen’s own Claude 4.5 column, does not pass Opus 4.5 on SWE-bench Verified or Terminal-Bench.

**DeepSeek-V4-Flash** is the small control, not the quality bet.

## Primary sources

Hardware

- https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html
- https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html
- https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/

Claude 4.5

- https://www.anthropic.com/news/claude-sonnet-4-5
- https://www.anthropic.com/news/claude-opus-4-5
- https://www.anthropic.com/news/claude-haiku-4-5
- https://www.anthropic.com/claude-sonnet-4-5-system-card

Models and recipes

- https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- https://huggingface.co/zai-org/GLM-5
- https://huggingface.co/zai-org/GLM-5.1
- https://huggingface.co/zai-org/GLM-5.2
- https://huggingface.co/zai-org/GLM-5.3-BF16
- https://huggingface.co/moonshotai/Kimi-K2.5
- https://huggingface.co/moonshotai/Kimi-K2.6
- https://huggingface.co/moonshotai/Kimi-K3
- https://recipes.vllm.ai/moonshotai/Kimi-K3
- https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/docs/deploy_guidance.md
- https://recipes.vllm.ai/moonshotai/Kimi-K2.5
- https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
- https://huggingface.co/MiniMaxAI/MiniMax-M2.7
- https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE
- https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

On-disk sizes are the `usedStorage` field of `https://huggingface.co/api/models/<repo>` as returned on 2026-10-09.
