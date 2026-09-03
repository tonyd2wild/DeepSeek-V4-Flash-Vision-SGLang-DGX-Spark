# DeepSeek-V4-Flash-Vision-Exp on NVIDIA DGX Spark with SGLang

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

Images go in as OpenAI `image_url` content on `/v1/chat/completions`; text requests work unchanged.

### Benchmarks (TP2)

Real prompts only, no counting prompts in the headline (40 prompts, 8 categories, `tools/bench_categories.py`). Prefill measured cold. C1 = one stream, C4 = four, C16 = sixteen concurrent.

| lane | prose tok/s | code tok/s | C4 aggregate | C16 aggregate | TTFT C1 | notes |
|---|---|---|---|---|---|---|
| SGLang TP2 (this repo) | pending | pending | pending | pending | pending | 327K ctx |
| vLLM DSpark TP2 (ours) | pending | pending | pending | pending | pending | 1M ctx |
| vLLM DSpark TP4 (ours, 2026-09-02) | 42 | 98 | | 124 | 0.90 s | 1M ctx, 8.33M-token KV |

## Lane 2: TP4 (4 Sparks)

Planned. The cookbook ships only the 2-node cell; the TP4 lane will reuse the same image and flags with `--tp 4 --nnodes 4` if the preview build allows it, and results land in the same table.

## Revert / coexistence

This recipe replaces GLM-5.3-Flash TP4 on the same nodes. The revert snapshot for that deployment lives with the launchers (`REVERT-TO-GLM-TP4-2026-09-03.md` in our launcher archive).

## Credits

SGLang team for the DGX Spark preview build and the verified cell (thanks to the SGLang contact who sent it ahead of the post). DeepSeek for the checkpoint. Fleet, launcher, NCCL tuning and benchmarks: tonyd2wild.
