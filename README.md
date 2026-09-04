# DeepSeek-V4-Flash-Vision-Exp on NVIDIA DGX Spark with SGLang

> **Status: staging checkpoint (2026-09-03).** First working TP2 deployment and first real-prompt numbers. Untuned: the DSpark accept length and prefill path have not been worked on yet, the TP4 lane is not built, and the like-for-like vLLM TP2 run is still owed. Numbers here are a baseline, not a verdict.

Two lanes, one repo: **TP2 (2 Sparks)** now, **TP4 (4 Sparks)** to follow. Same checkpoint, same engine, same benchmark harness, so the numbers are comparable lane to lane and against our vLLM DSpark recipe.

- Model: [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) (305B, 0731 Flash base + vision encoder, bundled DSpark draft head, native FP4-mixed weights)
- Engine: SGLang, preview image `lmsysorg/sglang:dev-v4f-2dgx-v2` (DGX Spark only; branch `b12x-vision` @ `452239a74f`). Source of the verified cell: the [SGLang DeepSeek-V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) (`dgx-spark / flash-vision / fp4 / balanced / multi-2`).
- Hardware: DGX Spark (GB10, 128 GB unified) x2, ConnectX-7 RoCE on one L2 (`192.168.192.0/24`), weights on the head node, worker reads them over NFS.

## Lane 1: TP2 (2 Sparks)

Status: **live 2026-09-03** (Bluey head + Asusi worker). Benchmarks below are filled from `results/`.

```
# worker first (rank 1), then head (rank 0)
bash launch/sglang-ds4v-tp2.sh 1     # on the worker
bash launch/sglang-ds4v-tp2.sh 0     # on the head; serves http://<head>:30000/v1
```

What the launcher does (see the file for every flag):
- `docker run --network host --ipc host --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband` so NCCL uses RDMA (TCP fallback costs about 40% decode per the cookbook).
- Cookbook env: `SGLANG_SM120_FLASHMLA_BACKEND=b12x`, `B12X_MLA_SM120_DSV4_H16_NATIVE=1`, `SGLANG_OPT_FUSE_MHC_POST_PRE=1`, `SGLANG_OPT_FP8_WO_A_GEMM=1`, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `SGLANG_B12X_MAX_TOKENS=8192` (must equal chunked prefill), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Our fleet NCCL settings for the GB10 pair: `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0`.
- Serve flags: `--tp 2 --nnodes 2 --node-rank N --dist-init-addr <head>:5000 --moe-runner-backend b12x --speculative-algorithm DSPARK --chunked-prefill-size 8192 --context-length 327680 --mem-fraction-static 0.80 --swa-full-tokens-ratio 0.2 --cuda-graph-max-bs-decode 32 --max-running-requests 32`.

### GB10 load-time memory (what the cookbook cell does not tell you)

The cell's `--mem-fraction-static 0.80` reserves 80% of the 128 GB unified memory up front, then the weight loader needs tens of GB of host RAM on top while it converts the FP8 shared experts into the FP4 fused MoE weights. On our nodes that killed the rank-0 scheduler (Linux OOM killer, `anon-rss 38 GB`). Lowering the fraction is not the answer: the loaded TP2 shard is about 73% of the node (SGLang: "minimum viable = 0.731"), so 0.72 loads and then exits with no room for KV. What works:

1. **A swapfile on each node** so the load-time transient spills instead of dying: `sudo fallocate -l 48G /swapfile-sglang && sudo chmod 600 /swapfile-sglang && sudo mkswap /swapfile-sglang && sudo swapon /swapfile-sglang`. Peak swap use seen during load: about 36 GB per node; it drains to near zero once serving.
2. Keep `--mem-fraction-static 0.80` (the launcher default). That leaves roughly 9 GB of KV per node at TP2.
3. The launcher tames the spike at the source: `--model-loader-extra-config '{"enable_multithread_load":false}'` (the image's default 8-thread loader keeps about ten 3.6 GB shards of host buffers in flight while the full parameter store is already resident), `--weight-loader-drop-cache-after-load` (per-shard page-cache release) and `--startup-weight-load-mode serial`. With those, swap is a safety net rather than a requirement.

Where the memory goes at TP2 (from the shard headers): about 84 GB of weights per node (73.6 GB routed experts, 5.4 GB DSpark draft, 3 GB attention, 1 GB embeddings and head, 0.9 GB vision encoder). KV cost is about 14.6 KB per token at `--swa-full-tokens-ratio 0.2` (9.3 KB at 0.1), replicated on both ranks, so 0.80 leaves roughly 0.5M tokens of KV per node. 32 concurrent requests at the full 327,680 context do not fit; SGLang caps running requests to the pool.

The launcher also mounts `/var/tmp/sglang-cache` to `/root/.cache` so the CuTeDSL and FlashInfer JIT and autotune results survive a relaunch (a cold start pays 10 to 15 minutes of compilation).

Images go in as OpenAI `image_url` content on `/v1/chat/completions`; text requests work unchanged.

### Benchmarks (TP2)

Real prompts only, no counting prompts in the headline (40 prompts, 8 categories, `tools/bench_categories.py`). Prefill measured cold. C1 = one stream, C4 = four, C16 = sixteen concurrent.

| lane | prose tok/s | code tok/s | C1 median (all 40) | C4 aggregate | C16 aggregate | TTFT C1 / C16 | auto score C1 / C16 | notes |
|---|---|---|---|---|---|---|---|---|
| SGLang TP2, this repo (2026-09-03) | 31.2 | 59.6 | 45.3 | 58.4 | 74.7 | 0.36 s / 1.42 s | 0.906 / 0.864 | 327K ctx, 507K-token KV pool, DSpark accept ~2.2 tok/step |
| vLLM DSpark TP4, ours (2026-09-02) | 42.0 | 98.3 | 74.2 | 94.2 | 123.9 | 0.21 s / 0.90 s | 0.873 / 0.894 | 1M ctx, 8.33M-token KV; fresh 1.6K-token prompt TTFT 0.94 s |
| vLLM DSpark TP2, ours | not yet run through this harness | | | | | | | 1M ctx, ~2.79M-token KV; counting-ladder c1 53 / c6 160 tok/s (2026-08-31) |

Per-category C1 decode tok/s (auto score): SGLang TP2 coding 59.6 (1.00), reasoning 51.3 (0.80), json 71.7 (1.00), html 73.6 (0.80), prose 31.2 (0.80), narrative 27.6 (0.90), summary 37.7 (0.95), format 28.5 (1.00). vLLM TP4: coding 98.3 (1.00), reasoning 88.3 (0.80), json 85.4 (1.00), html 103.4 (0.89), prose 42.0 (0.60), narrative 44.8 (0.90), summary 61.3 (0.80), format 50.3 (1.00).

Read: on these Sparks the SGLang preview decodes at about 60% of our vLLM TP4 lane single-stream and scales less under concurrency (16 streams reached 75 tok/s aggregate, 13 tok/s per stream). Quality is equal within noise on the same prompts at temperature 0. The DSpark accept length observed during the runs was 1.7 to 2.6 tokens per step versus the ~3.2 the cookbook quotes for Vision. Raw files: `results/categories_sglang_tp2_off_c{1,4,16}.json`, `results/summary.md`.

### Ceiling ladder (counting prompt, not a headline number)

`tools/bench_sweep.py` runs "list the numbers 1 to 300" at C1 to C6. That prompt maximizes draft acceptance and is reported only as a peak, at the bottom, never as decode speed. See `results/sweep_sglang_tp2.json` and `results/sweep_ds4tp4.json`.

| c | SGLang TP2 aggregate tok/s | per stream | TTFT | vLLM TP4 aggregate | per stream | TTFT |
|---|---|---|---|---|---|---|
| 1 | 78.6 | 78.7 | 1.05 s | 95.2 | 95.2 | 0.33 s |
| 2 | 132.1 | 66.1 | 0.82 s | 171.0 | 85.5 | 0.32 s |
| 3 | 177.9 | 60.3 | 0.86 s | 214.9 | 73.7 | 0.37 s |
| 4 | 224.6 | 56.6 | 0.90 s | 290.2 | 74.0 | 0.69 s |
| 5 | 264.6 | 53.8 | 0.91 s | 266.3 | 54.2 | 0.58 s |
| 6 | 291.9 | 49.6 | 1.23 s | 367.9 | 62.5 | 0.66 s |

Peak max acceptance: SGLang TP2 reaches 292 tok/s aggregate at c6 on this prompt (vLLM TP4 368). Cold prefill on a ~1.5K-token prompt at c1: 1,543 tok/s (SGLang TP2) versus 4,868 tok/s (vLLM TP4).

## Lane 2: TP4 (4 Sparks)

Planned. The cookbook ships only the 2-node cell; the TP4 lane will reuse the same image and flags with `--tp 4 --nnodes 4` if the preview build allows it, and results land in the same table.

## Revert / coexistence

This recipe replaces GLM-5.3-Flash TP4 on the same nodes. The revert snapshot for that deployment lives with the launchers (`REVERT-TO-GLM-TP4-2026-09-03.md` in our launcher archive).

## Credits

SGLang team for the DGX Spark preview build and the verified cell (thanks to the SGLang contact who sent it ahead of the post). DeepSeek for the checkpoint. Fleet, launcher, NCCL tuning and benchmarks: tonyd2wild.
