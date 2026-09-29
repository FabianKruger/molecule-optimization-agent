# Local LLM scripts

Settled 2026-09-29. Do not implement a later session in the same change as an earlier one.

The harness is not modified. Experiment YAMLs stay as committed. You set `llm.model` yourself to the profile's served name before a run. Both scripts assume they are already inside `salloc` or `sbatch`. They do not call `sbatch` and do not know the partition or account.

## Profiles

`--profile` overrides `MOLOPT_LLM_PROFILE`. If neither is set, the script exits. There is no default profile.

| Profile | Hugging Face id and `llm.model` | vLLM GPUs | Boltz GPU | Context |
| --- | --- | --- | --- | --- |
| `qwen3-32b` | `Qwen/Qwen3-32B-FP8` | `0` | `1` | 32768 (native). No YaRN. |
| `deepseek-v3.2` | `deepseek-ai/DeepSeek-V3.2` | `0,1,2,3` | `4` | 131072. Requests past that fail. |

One module owns this table (`scripts/local_llm/profiles.py`). The download script and the bootstrap script both import it.

DeepSeek serve flags (tensor parallel 4, FP8, FlashInfer MoE) are filled in when the bootstrap session is implemented, from the vLLM 0.30 recipe for that checkpoint. They are not copied from a blog post.

## Shared rules

- `--weights-dir` is required and must be an absolute path. It is `HF_HOME`. If the path is inside `$HOME`, exit. Do not download onto NFS home.
- vLLM listens on `127.0.0.1` only. Default port `8000` (`--port` overrides).
- Request logging stays off. Prompts must not land in the vLLM log.
- The parent process does not export `CUDA_VISIBLE_DEVICES`. Each child gets its own value.
- Boltz-2 inherits the environment of the `molopt` process. That process gets only the profile's Boltz GPU.
- `OPENAI_BASE_URL` is set to the local `/v1` URL for the `molopt` process. If `OPENAI_API_KEY` is unset, set it to `local`. The harness refuses to start without a key. Do not clear a key that is already set.
- Teardown kills the vLLM process group on success, on `molopt` failure, on health timeout, and on signals. The script's exit code is the experiment's exit code when the experiment ran. If vLLM dies first, stop `molopt` and exit non-zero.

## Session 1 — download

`scripts/local_llm/download.py`

- Requires `--profile` (or `MOLOPT_LLM_PROFILE`) and `--weights-dir`.
- Downloads that profile's snapshot with the `vllm` pixi environment's Hugging Face client.
- Idempotent. A snapshot that is already complete exits 0.
- Does not start vLLM. Does not need GPUs.
- Does not implement `--load-check` beyond accepting the flag and exiting with "not implemented" if it is passed. Session 2 fills it in.

## Session 2 — load check

Same download script. `--load-check` is off by default.

- When set, start `vllm serve` for that profile against `--weights-dir`, on that profile's GPUs, bound to localhost.
- Poll `/health`. On success, kill the server and exit 0.
- On timeout or server death, kill the server and exit non-zero.
- Health timeout is part of the profile: 10 minutes for `qwen3-32b`, 45 minutes for `deepseek-v3.2`.
- This flag needs the accelerators. The default download path still does not.

## Session 3 — bootstrap

`scripts/local_llm/bootstrap.py`

One experiment per invocation.

1. Resolve profile and `--weights-dir` the same way as the download script.
2. Require `--config`. Read `llm.model`. If it is not the profile's served name, exit before starting vLLM.
3. Start vLLM with the profile's GPUs and `HF_HOME=--weights-dir`.
4. Wait for `/health` up to the profile's timeout.
5. Run `pixi run molopt --config <path>` from the repo root, in the default pixi environment, with only the Boltz GPU visible.
6. Tear down vLLM. Exit with `molopt`'s status.

Out of scope for this session: a campaign of several configs, rewriting YAML, Slurm submission, and changing the harness.

## Later, not a coding session

Run `qwen3-32b` once, with `llm.model` set to `Qwen/Qwen3-32B-FP8`, before any DeepSeek allocation. That run is the proof that the port, the key, the model-name check, and Boltz-on-another-GPU line up.
