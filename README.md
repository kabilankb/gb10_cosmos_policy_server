# gb10_cosmos_policy_server

Docker setup for running the Cosmos3 action-policy server (`nvidia/Cosmos3-Nano-Policy-DROID`) against the RoboLab benchmark harness on a GB10 (Grace Blackwell, unified-memory, DGX Spark-class) box — aarch64, CUDA 13.0, compute capability 12.1.

This repo vendors the `Dockerfile`/entrypoint, the GB10-specific source patches, and the policy server script needed on top of an `NVIDIA/cosmos-framework` checkout. It is not a fork of `cosmos-framework` — apply `patches/` to your own checkout before building.

## 0. Apply the GB10 patches

From the root of your `cosmos-framework` checkout:

```shell
cp /path/to/gb10_cosmos_policy_server/patches/sdpa_fallback.py \
   cosmos_framework/model/attention/sdpa_fallback.py

git apply /path/to/gb10_cosmos_policy_server/patches/backends.py.diff
git apply /path/to/gb10_cosmos_policy_server/patches/frontend.py.diff
git apply /path/to/gb10_cosmos_policy_server/patches/vision.py.diff
```

These are required before building on GB10 — neither has a `docker run` flag or env var workaround:

- **No compiled attention kernel targets arch_tag 121.** `flash2`/`flash3`/`natten` all lack kernels for GB10's compute capability. `sdpa_fallback.py` adds a plain PyTorch `scaled_dot_product_attention` backend, registered in `backends.py`/`frontend.py` as the fallback.
- **`torchvision.io.read_video` was removed** in the torchvision version GB10's CUDA-13.0 PyTorch build ships. `vision.py.diff` replaces it with a manual PyAV decode loop in `read_media()`.

Copy this repo's `Dockerfile` and `docker/entrypoint.sh` into that checkout (or build with `-f` pointed at this repo) before step 1.

## 1. Build the Docker image

```shell
docker build \
  -t cosmos-framework:latest \
  .
```

## 2. Set your Hugging Face token and launch the container

Launching installs the dependencies inside the running container.

```shell
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

**Setup problem — `HF_HOME` silently shadows the bind-mounted cache.** This command sets `HF_HOME=/workspace/.cache/huggingface` *and* bind-mounts `$HOME/.cache/huggingface:/root/.cache/huggingface` — but `HF_HOME` wins, so anything already cached at the mounted path is ignored. A 2.82 GB checkpoint already sitting on the host got re-downloaded in full for this reason. If you want the container to reuse an existing host-side HF cache, drop the `HF_HOME` override or point it at `/root/.cache/huggingface` instead.

## 3. Start the policy server

`scripts/action_policy_server_robolab.py` in this repo is the same script the container runs; copy it into `cosmos_framework/scripts/` in your checkout if it's not already there.

```shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000
```

**Setup problem — the checkpoint download looks stuck but isn't.** The first launch downloads the ~33 GB `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint via `hf download --format=json`, which disables the progress bar to emit one clean JSON blob at the end — so the terminal shows **zero output** for the entire transfer. This is not a hang. Verify from a second shell into the same container:

```shell
docker exec <container> bash -c "ps aux | grep 'hf download' | grep -v grep"
docker exec <container> bash -c "du -sb /workspace/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano-Policy-DROID"
```

**Setup problem — NVML can't report memory on unified-memory GPUs.** On GB10, the server auto-sizes parallelism via `pynvml.nvmlDeviceGetMemoryInfo()`, which raises `NVMLError_NotSupported` because there's no discrete VRAM framebuffer to query (even `nvidia-smi` on the host reports `Memory-Usage: Not Supported`). Fix: pass `--device-memory-bytes` explicitly.

**Setup problem — guardrails are on by default and gated.** Guardrails require downloading `nvidia/Cosmos-Guardrail1`, a gated Hugging Face repo; if the account behind `HF_TOKEN` lacks approved access, the server crashes on boot with `Error: Access denied. This repository requires approval.` Fix: request access, or pass `--no-guardrails` for local dev.

```shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000 \
  --device-memory-bytes $(free -b | awk '/^Mem:/{print $2}') \
  --no-guardrails
```

## 4. Client

Health check, from a second shell:

```shell
curl http://localhost:8000/healthz
```

Run a RoboLab task against the server:

```shell
python policies/cosmos3/run.py --task BananaInBowlTask
```

Nothing about this is banana-specific — `--task <TaskName>` runs any of RoboLab's 120 registered tasks (64 simple / 39 moderate / 17 complex, `robolab/tasks/_metadata/task_table.csv`) against the same running server. `BananaInBowlTask` is `simple` difficulty with a single `semantics` attribute (pick up one named object, place it in one named container); 26 other tasks share that identical profile:

- **Nearly identical shape** — `BagelsOnPlateTask`, `WoodSpatulaToBowlTask`, `FruitsOnionTask`, `MoveBananaToBagelPlateTask`
- **Same "X in bin" pattern** — `CannedFoodInBinTask`, `CoffeePotInBinTask`, `SmartphoneInBinTask`, `AnimalsInBinTask`, `ToyInBinTask`, `ClutterPumpkinTask`
- **Other single-object grab/place** — `GrabABagelTask`, `GrabAFruitTask`, `PickDrillTask`, `PickGlassesTask`, `MouseOnKeyboardTask`, `OneBottleOnShelfTask`, `OneBottleInSquarePailTask`, `JugsOnShelfTask`, `RecycleCartonsOnBoxTask`, `ThrowAwayAppleTask`, `ThrowAwaySnacksTask`, `TakeMeasuringSpoonOutTask`, `FoodPacking1BoxesTask`, `FoodPacking1CansTask`, `FoodPacking2BoxesTask`, `FoodPacking2CansTask`
- **Directly banana-themed** — `BananaOnPlateTask` (same object, plate instead of bowl, no attributes); `BananaThenRubiksCubeTask` / `RubiksCubeAndBananaTask` / `RubiksCubeOrBananaTask` / `RubiksCubeThenBananaTask` (banana + cube, adds a `conjunction` attribute); `BananasInBinOneMoreTask` / `BananasInBinThreeTotalTask` / `BananasInCrateTask` / `BananasOutOfBinTask` (moderate difficulty, adds counting)

`BananaOnPlateTask` is the closest direct comparison to run next — same object, same difficulty, just a swapped container.

## 5. Fine-tune process

Fine-tuning a new embodiment onto `Cosmos3-Nano` runs in the same container, using the `mot_fsdp` + `PackingDataLoader` + `RankPartitionedDataLoader` stack.

**Stage the dataset and convert the base checkpoint to DCP:**

```shell
uvx hf@latest download --repo-type dataset <org>/<dataset> --local-dir examples/data/<dataset>

python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano --checkpoint-path Cosmos3-Nano
```

**Validate the config with a dry run** before spending GPU time:

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml=<path/to/config>.toml --dryrun
```

**Launch training** via `torchrun`, driven by a structured TOML (`[job]`, `[model.parallelism]`, `[optimizer]`, `[trainer]`, `[checkpoint]` sections):

```shell
bash examples/launch_sft_action_policy_so101_nano.sh
```

**Export the trained checkpoint and serve it** through the same policy server used in step 3:

```shell
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$RUN_DIR/checkpoints/<iter>" \
  --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

### GPU requirement

**A single 8-GPU node.** The reference recipe pins `data_parallel_shard_degree = 8` (FSDP shard across all 8 GPUs on one node, `data_parallel_replicate_degree = 1`), global batch 256 (32 samples per rank × 8 shards), lr 5e-5, 2000 iterations, checkpointing every 500 iters. That's a LIBERO-10-scale starting point for a dataset in the low thousands of episodes, not an empirically tuned reference number — expect to retune `trainer.max_iter` / `optimizer.lr` against the first run's loss curve.

To scale beyond one node without changing the per-rank batch, raise `model.parallelism.data_parallel_replicate_degree` (HSDP) — e.g. `--opts model.parallelism.data_parallel_replicate_degree=4` for 4 nodes — or raise `grad_accum_iter` on a single node instead of adding hardware.
