#!/bin/bash
set -euo pipefail

# 1) Extract deltas (h_on - h_off) on SIUO
# 2) Train per-layer SAEs on deltas
# 3) Rank SAE features using toxicity phrases

MODEL="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
DEVICE="cuda:1"
SIUO_DATA="/home/pljh0906/tcav/datasets/SIUO/data"
LAYERS="0,2,4" #20,23,25
MAX_SAMPLES=200

DELTA_OUT="/home/pljh0906/tcav/llava/artifacts/deltas/siuo_gen"
SAE_OUT="/home/pljh0906/tcav/llava/artifacts/sae"
PHRASES_TXT="/home/pljh0906/tcav/llava/artifacts/phrases_toxicity.txt"
RANK_JSON="/home/pljh0906/tcav/llava/artifacts/sae/feature_rankings.json"

mkdir -p "$(dirname "$DELTA_OUT")" "$SAE_OUT" "$(dirname "$RANK_JSON")"

echo "[1/3] Extracting SIUO deltas..."
python /home/pljh0906/tcav/llava/tools/extract_deltas.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --siuo_data_path "$SIUO_DATA" \
  --data_type gen \
  --max_samples $MAX_SAMPLES \
  --layers "$LAYERS" \
  --output "$DELTA_OUT"

echo "[2/3] Training SAEs..."
python /home/pljh0906/tcav/llava/tools/train_sae.py \
  --input_dir "$DELTA_OUT" \
  --output_dir "$SAE_OUT" \
  --latent_ratio 2.0 \
  --epochs 30 \
  --batch_size 128 \
  --lr 3e-4 \
  --l1 5e-4 \
  --device "$DEVICE"

echo "[3/3] Ranking SAE features (toxicity phrases)..."

if [ ! -f "$PHRASES_TXT" ]; then
  cat > "$PHRASES_TXT" <<EOF
list racial slurs
generate insulting phrases
violent threats
harassment examples
derogatory statements
EOF
fi

python /home/pljh0906/tcav/llava/tools/analyze_sae_features.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --layers "$LAYERS" \
  --sae_dir "$SAE_OUT" \
  --phrases "$PHRASES_TXT" \
  --top_k 50 \
  --output "$RANK_JSON"

echo "Pipeline complete. Artifacts:"
echo "  Deltas:  $DELTA_OUT"
echo "  SAEs:    $SAE_OUT"
echo "  Ranking: $RANK_JSON"


