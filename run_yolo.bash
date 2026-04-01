#!/bin/bash
# ============================================================
# Adversarial Attack Experiments on YOLOv8 (Argoverse)
# ============================================================
# Model   : YOLOv8 Nano (ckpt/yolov8n.pt)
# Dataset : data/Argoverse_sample1_amnesia
# Results : result/[attack_method]/yolov8n/[date]/
# ============================================================

MODEL_TYPE="yolov8"
CHECKPOINT="ckpt/yolov8n.pt"
IMAGE_DIR_ARGO="data/Argoverse_sample1_amnesia"
ANN_FILE_ARGO="data/Argoverse_sample1_amnesia/labels/val"

IMAGE_DIR_COCO="data/coco_amnesia/val2017"
ANN_FILE_COCO="data/coco_amnesia/instances_val2017_ori.json"

MAX_QUERY=1000
SCORE_THR=0.3
IOU_THR=0.5
SUCCESS_THR=0.7
SEED=42
LOG_INTERVAL=50
NUM_IMAGES=2

run_yolo_attacks() {
  local ds_name=$1
  local img_dir=$2
  local ann_file=$3

  echo "=========================================="
  echo ">>> Dataset: $ds_name"
  echo "=========================================="

  echo "[1/2] Running SparseEvo Attack on YOLOv8 ($ds_name)..."
  python run_attack.py \
    --model-type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --image-dir $img_dir \
    --ann-file $ann_file \
    --attack sparse_evo \
    --dataset-name "$ds_name" \
    --max-query $MAX_QUERY \
    --score-thr $SCORE_THR \
    --iou-thr $IOU_THR \
    --success-thr $SUCCESS_THR \
    --seed $SEED \
    --num-images $NUM_IMAGES \
    --pop-size 10 \
    --cr 0.9 \
    --mu 0.01

  echo "[2/2] Running PointWise Multi Scheduling Attack on YOLOv8 ($ds_name)..."
  python run_attack.py \
    --model-type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --image-dir $img_dir \
    --ann-file $ann_file \
    --attack pointwise_multi_sched \
    --dataset-name "$ds_name" \
    --max-query $MAX_QUERY \
    --score-thr $SCORE_THR \
    --iou-thr $IOU_THR \
    --success-thr $SUCCESS_THR \
    --seed $SEED \
    --num-images $NUM_IMAGES \
    --npix 0.1
}

# 1. Run on Argoverse
run_yolo_attacks "Argoverse" $IMAGE_DIR_ARGO $ANN_FILE_ARGO

# 2. Run on COCO Amnesia
run_yolo_attacks "COCO" $IMAGE_DIR_COCO $ANN_FILE_COCO

echo "=========================================="
echo "All YOLOv8 attacks completed!"
echo "=========================================="
