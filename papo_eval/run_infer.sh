#!/usr/bin/env bash
set -euo pipefail

# which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

########################  Data  ########################
DATASET="hiyouga_geometry3k"
# DATASET="AI4Math_MathVerse"
# DATASET="AI4Math_MathVista"
# DATASET="We-Math_We-Math"
# DATASET="MMMU_MMMU_Pro"
# DATASET="BUAADreamer_clevr_count_70k"
# DATASET="AI4Math_MathVerse_vision_dependent"
# DATASET="lscpku_LogicVista"
# DATASET="PAPO_MMK12"

########################  Model  ########################
MODEL_NAME="PAPO-G-H-Qwen2.5-VL-3B"
MODEL="PAPOGalaxy/${MODEL_NAME}"

########################  Config  ########################
SAVE_PATH="infer_outputs/$MODEL_NAME/$DATASET.jsonl"
ROLLOUT_N=8

########################  Run  ########################
echo "Running inference for dataset: $DATASET → $SAVE_PATH"

python ./papo_eval/run_infer.py \
  --model_name_or_path "$MODEL" \
  --template qwen2_vl \
  --cutoff_len 9216 \
  --max_new_tokens 2048 \
  --temperature 1.0 \
  --top_p 1.0 \
  --batch_size 64 \
  --save_every 10 \
  --save_name "$SAVE_PATH" \
  --dataset "$DATASET" \
  --loop_n $ROLLOUT_N