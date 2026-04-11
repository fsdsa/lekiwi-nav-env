#!/bin/bash
# vLLM server for Qwen3-VL-8B-Instruct (A100 서버)
#
# VLM + VLA 동시 추론을 위한 GPU 메모리 분배:
#   VLM (vLLM, Qwen3-VL-8B bf16): --gpu-memory-utilization 0.75 → ~29.8GB
#   VLA (Pi0-FAST):                ~8.1GB
#   합계: ~37.9GB / A100 40GB
#
# 사용법 (서버에서 직접):
#   conda activate vllm
#   bash run_vllm_server.sh
#
# VLA 서버도 함께 띄우려면:
#   bash launch_servers.sh
#
# 클라이언트 테스트 (로컬):
#   python vlm_client.py --server http://218.148.55.186:8000

export PATH="/home/jovyan/miniconda3/envs/vllm/bin:$PATH"

exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code
