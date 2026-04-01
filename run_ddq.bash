#!/bin/bash
# ============================================================
# Adversarial Attack Experiments on DDQ-DETR (COCO Amnesia)
# ============================================================
# Model   : DDQ-DETR-4scale R50
# Dataset : data/coco_amnesia/val2017 (100 images)
# Results : result/[attack_method]/[model]/[date]/
# ============================================================

# num_images=2
CONFIG=configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py
CHECKPOINT=ckpt/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth
IMAGE_DIR=data/coco_amnesia/val2017
ANN_FILE=data/coco_amnesia/instances_val2017_ori.json
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
echo "[1/2] Running SparseEvo Attack..."
echo "=========================================="
python run_attack.py \
  --config $CONFIG \
  --checkpoint $CHECKPOINT \
  --image-dir $IMAGE_DIR \
  --ann-file $ANN_FILE \
  --attack sparse_evo \
  --dataset-name "COCO_Amnesia" \
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
echo "[2/2] Running PointWise Multi Scheduling Attack..."
echo "=========================================="
python run_attack.py \
  --config $CONFIG \
  --checkpoint $CHECKPOINT \
  --image-dir $IMAGE_DIR \
  --ann-file $ANN_FILE \
  --attack pointwise_multi_sched \
  --dataset-name "COCO_Amnesia" \
  --max-query $MAX_QUERY \
  --score-thr $SCORE_THR \
  --iou-thr $IOU_THR \
  --success-thr $SUCCESS_THR \
  --seed $SEED \
  --npix 0.1

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: result/"
echo "=========================================="
