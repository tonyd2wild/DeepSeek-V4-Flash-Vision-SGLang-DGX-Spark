#!/usr/bin/env bash
# SGLang DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark (TP=2 over RoCE), from the SGLang cookbook's
# verified "dgx-spark / flash-vision / fp4 / balanced / multi-2" cell, with our fleet's NCCL settings.
# Usage: bash sglang-ds4v-tp2.sh <rank>    rank 0 = head (Bluey 192.168.192.1, local weights)
#                                           rank 1 = worker (Asusi 192.168.192.3, weights over NFS from Bluey)
# Start the worker first, then the head. Serves OpenAI API on the head at :30000.
set -euo pipefail
RANK="${1:?rank 0|1}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-v4f-2dgx-v2}"
MODEL_DIR="${MODEL_DIR:-DeepSeek-V4-Flash-Vision-Exp}"
NAME="${NAME:-sglang_ds4v}"
HEAD_IP="192.168.192.1"; DIST_PORT="${DIST_PORT:-5000}"; PORT="${PORT:-30000}"
CTX="${CTX:-327680}"; MAX_REQ="${MAX_REQ:-32}"; CG_BS="${CG_BS:-32}"
# The cookbook cell says --mem-fraction-static 0.80. On our GB10s that OOM-killed the rank-0 scheduler during
# weight load (loader peaked at 38 GB of host RAM while converting FP8 shared experts to FP4). 0.72 plus a
# page-cache flusher during load (see README) loads cleanly. Override with MEM_FRAC=0.80 to reproduce the cell.
MEM_FRAC="${MEM_FRAC:-0.72}"
# Loader staging is the other half of the load-time spike: the default multi-thread shard loader holds several
# 3.5 GB shards in host RAM at once. Two threads plus a swapfile (48 GB, see README) rides through it.
LOADER_THREADS="${LOADER_THREADS:-2}"
SGLANG_EXTRA="${SGLANG_EXTRA:-} --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":$LOADER_THREADS}"
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
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/models/$MODEL_DIR:ro" \
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
    --host 0.0.0.0 --port "$PORT" ${SGLANG_EXTRA:-}
echo "started $NAME rank $RANK on $(hostname); logs: docker logs -f $NAME"
