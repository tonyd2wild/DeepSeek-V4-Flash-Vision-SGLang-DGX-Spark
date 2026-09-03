#!/usr/bin/env bash
# SGLang DeepSeek-V4-Flash-Vision-Exp on 2x NVIDIA DGX Spark (TP=2 over RoCE), from the SGLang cookbook's
# verified "dgx-spark / flash-vision / fp4 / balanced / multi-2" cell, with our fleet's NCCL settings.
#
# Usage: bash sglang-ds4v-tp2.sh <rank>     rank 0 = head (serves the OpenAI API on :30000), rank 1 = worker
# Start the worker first, then the head. Defaults below are our fleet's; override with env vars:
#   HEAD_IP (192.168.192.1)  DIST_PORT (5000)  PORT (30000)  MODEL_DIR  MODEL_HOST  IMAGE
#   MEM_FRAC (0.72)  LOADER_THREADS (unset = image default)  CTX (327680)  MAX_REQ (32)  CG_BS (32)
#   CACHE_HOST (/var/tmp/sglang-cache, persists the CuTeDSL/FlashInfer JIT + autotune across relaunches)
#   SGLANG_EXTRA (extra serve args, space separated)
set -euo pipefail
RANK="${1:?rank 0|1}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-v4f-2dgx-v2}"
MODEL_DIR="${MODEL_DIR:-DeepSeek-V4-Flash-Vision-Exp}"
NAME="${NAME:-sglang_ds4v}"
HEAD_IP="${HEAD_IP:-192.168.192.1}"; DIST_PORT="${DIST_PORT:-5000}"; PORT="${PORT:-30000}"
CTX="${CTX:-327680}"; MAX_REQ="${MAX_REQ:-32}"; CG_BS="${CG_BS:-32}"
CACHE_HOST="${CACHE_HOST:-/var/tmp/sglang-cache}"
# --mem-fraction-static 0.80 is the cookbook value and the floor that works: the loaded TP2 weight shard is ~73%
# of a GB10's 128 GB (SGLang reports "minimum viable 0.731"), so 0.80 leaves ~9 GB of KV per node. Going lower
# does not help; the load-time OOM on GB10 is the loader's host-RAM spike, which a swapfile absorbs (see README).
MEM_FRAC="${MEM_FRAC:-0.80}"
case "$RANK" in
  0) MODEL_HOST="${MODEL_HOST:-/var/tmp/models/$MODEL_DIR}" ;;
  1) # worker: local copy if it has one, else the head's NFS export
     if [ -z "${MODEL_HOST:-}" ]; then
       if [ -f "/var/tmp/models/$MODEL_DIR/config.json" ]; then MODEL_HOST="/var/tmp/models/$MODEL_DIR"
       else MODEL_HOST="/mnt/bluey-models/$MODEL_DIR"; fi
     fi ;;
  *) echo "rank must be 0 or 1"; exit 2 ;;
esac
[ -f "$MODEL_HOST/config.json" ] || { echo "model not found at $MODEL_HOST"; exit 3; }
if [ "$RANK" = "0" ] && ! ip -br addr 2>/dev/null | grep -q "$HEAD_IP/"; then
  echo "rank 0 must run on the node that owns HEAD_IP=$HEAD_IP (this host does not)"; exit 4
fi
mkdir -p "$CACHE_HOST"
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true

# Thinking off by default so decode numbers are answer tokens (SGLang reads the "thinking" template kwarg for
# DeepSeek V4; the vLLM lane used the same default). Reasoning parser keeps any thinking out of `content`.
# Served id matches the HF repo so clients can use it as the model name.
EXTRA=(--default-chat-template-kwargs '{"thinking":false}' --reasoning-parser deepseek-v4
       --served-model-name "${SERVED_NAME:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}")
# Load-time host RAM on GB10: the image's default loader runs 8 threads and keeps ~10 shards (3.6 GB each) of
# host buffers in flight while the full 78 GiB parameter store is already resident, which is the ~38 GB spike
# that gets the scheduler OOM-killed. Single-thread mmap loading (page-cache backed), dropping each shard's
# cache after use, and serial startup keep the spike small. Set LOADER_THREADS=N to trade RAM for speed.
if [ -n "${LOADER_THREADS:-}" ]; then
  EXTRA+=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LOADER_THREADS}")
else
  EXTRA+=(--model-loader-extra-config '{"enable_multithread_load":false}')
fi
EXTRA+=(--weight-loader-drop-cache-after-load --startup-weight-load-mode serial)
if [ -n "${SGLANG_EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA+=($SGLANG_EXTRA)
fi

docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/models/$MODEL_DIR:ro" \
  -v "$CACHE_HOST:/root/.cache" \
  -e SGLANG_SM120_FLASHMLA_BACKEND=b12x -e B12X_MLA_SM120_DSV4_H16_NATIVE=1 \
  -e SGLANG_OPT_FUSE_MHC_POST_PRE=1 -e SGLANG_OPT_FP8_WO_A_GEMM=1 \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 -e SGLANG_B12X_MAX_TOKENS=8192 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  "$IMAGE" \
  sglang serve --trust-remote-code --model-path "/models/$MODEL_DIR" \
    --tp 2 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$HEAD_IP:$DIST_PORT" \
    --moe-runner-backend b12x --speculative-algorithm DSPARK \
    --chunked-prefill-size 8192 --context-length "$CTX" --mem-fraction-static "$MEM_FRAC" \
    --swa-full-tokens-ratio 0.2 --cuda-graph-max-bs-decode "$CG_BS" --max-running-requests "$MAX_REQ" \
    --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}"
echo "started $NAME rank $RANK on $(hostname); logs: docker logs -f $NAME"
