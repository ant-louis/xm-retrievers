#!/bin/bash

# To perform.
DO_TRAIN="--do_train"
DO_TEST="" #"--do_test"

# Dataset.
DATA="wikipedia" #"minipile"

# Models.
TEACHER="BAAI/bge-base-en-v1.5"
STUDENT="facebook/xmod-base"
MAX_SEQ_LEN=256
POOL="mean"

# Training.
TRAIN_LANG="en"
BATCH_SIZE=128
EPOCHS=20
OPTIMIZER="AdamW"
WD=0.01
LR=2e-5
SCHEDULER="warmuplinear"
WARMUP_RATIO=0.1
FP16="" #"--use_fp16"
BF16="--use_bf16"
SEED=42
LOG_STEPS=1
SAVE_CKPTS_DURING_TRAINING="--save_during_training"
OUTPUT="output/training"

# Evaluation.
TEST_LANG="de" #mMARCO: ("ar" "de" "en" "es" "fr" "hi" "id" "it" "ja" "nl" "pt" "ru" "vi" "zh")
SIM="cos_sim"
LOG_STEPS=1

# Run.
python src/retrievers/single_vector_distiller.py \
    $DO_TRAIN \
    --dataset "$DATA" \
    --train_lang "$TRAIN_LANG" \
    --student_model_name "$STUDENT" \
    --teacher_model_name "$TEACHER" \
    --max_seq_length $MAX_SEQ_LEN \
    --pooling "$POOL" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
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
    --test_lang "$TEST_LANG" \
    --sim "$SIM" \
    --output_dir "$OUTPUT"