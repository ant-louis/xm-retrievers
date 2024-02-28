#!/bin/bash

# To perform.
DO_TRAIN="--do_train"
DO_TEST="" #"--do_test"

# Dataset.
DATA="mmarco"
LANG="en"

# Model.
MODEL="facebook/xmod-base"
MAX_SEQ_LEN=128
POOL="mean"
SIM="cos_sim"

# Training.
BATCH_SIZE=128
EPOCHS=20
OPTIMIZER="AdamW"
LR=2e-5
SCHEDULER="warmuplinear"
WD=0.01
WARMUP_RATIO=0.1
FP16="" #"--use_fp16"
BF16="--use_bf16"
SEED=42
LOG_STEPS=1
SAVE_CKPTS_DURING_TRAINING="--save_during_training"
OUTPUT="output/training"

# Run.
python src/retrievers/single_vector_biencoder.py \
    --dataset "$DATA" \
    --language "$LANG" \
    --model_name "$MODEL" \
    --max_seq_length $MAX_SEQ_LEN \
    --pooling "$POOL" \
    --sim "$SIM" \
    $DO_TRAIN \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --optimizer "$OPTIMIZER" \
    --lr $LR \
    --wd $WD \
    --scheduler "$SCHEDULER" \
    --warmup_ratio $WARMUP_RATIO \
    $FP16 \
    $BF16 \
    --seed $SEED \
    $SAVE_CKPTS_DURING_TRAINING \
    --log_steps $LOG_STEPS \
    $DO_TEST \
    --output_dir "$OUTPUT"