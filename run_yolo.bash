#!/bin/bash
# ============================================================
# Adversarial Attack Experiments on YOLOv8 (Argoverse)
# ============================================================
# Model   : YOLOv8 Nano (yolov8n.pt) - Auto downloaded
# Dataset : data/Argoverse_sample1_amnesia
# Results : result/[attack_method]/yolov8n/[date]/
# ============================================================

MODEL_TYPE="yolov8"
CHECKPOINT="yolov8n.pt"
IMAGE_DIR="data/Argoverse_sample1_amnesia"
ANN_FILE="data/Argoverse_sample1_amnesia/labels/val"
MAX_QUERY=1000
SCORE_THR=0.3
IOU_THR=0.5
SUCCESS_THR=0.7
SEED=42
LOG_INTERVAL=50

# ============================================================
# 1. SparseEvo Attack
# ============================================================
echo "=========================================="
echo "[1/2] Running SparseEvo Attack on YOLOv8..."
echo "=========================================="
python run_attack.py \
  --model-type $MODEL_TYPE \
  --checkpoint $CHECKPOINT \
  --image-dir $IMAGE_DIR \
  --ann-file $ANN_FILE \
  --attack sparse_evo \
  --max-query $MAX_QUERY \
  --score-thr $SCORE_THR \
  --iou-thr $IOU_THR \
  --success-thr $SUCCESS_THR \
  --seed $SEED \
  --pop-size 10 \
  --cr 0.9 \
  --mu 0.01

# ============================================================
# 2. PointWise Multi Attack with Scheduling
# ============================================================
echo "=========================================="
echo "[2/2] Running PointWise Multi Scheduling Attack on YOLOv8..."
echo "=========================================="
python run_attack.py \
  --model-type $MODEL_TYPE \
  --checkpoint $CHECKPOINT \
  --image-dir $IMAGE_DIR \
  --ann-file $ANN_FILE \
  --attack pointwise_multi_sched \
  --max-query $MAX_QUERY \
  --score-thr $SCORE_THR \
  --iou-thr $IOU_THR \
  --success-thr $SUCCESS_THR \
  --seed $SEED \
  --npix 0.1

echo "=========================================="
echo "All YOLOv8 attacks completed!"
echo "=========================================="
