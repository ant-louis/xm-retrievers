#!/bin/bash

# To perform.
DO_TRAIN="--do_train"
DO_TEST="" #"--do_test"

# Data.
DATA="mmarco" #"mrtydi"
LANG="en"     #mMARCO: ("en" "fr" "ar" "de" "es" "hi" "id" "it" "ja" "nl" "pt" "ru" "vi" "zh"); Mr. TyDi: ("ar" "bn" "en" "fi" "id" "ja" "ko" "ru" "sw" "te")
DATA_DIR="data"

# Model.
MODEL="facebook/xmod-base"
DIM=128
SIM="cosine"
DOC_MAXLEN=256
QUERY_MAXLEN=32
MASK_PUNCT_IN_DOCS="--mask_punctuation"
ATTEND_MASK_TOKENS_IN_QUERIES="--attend_to_mask_tokens"

# Training.
TRAINING_TYPE="v1.5"
if [[ "$TRAINING_TYPE" == "v1" || "$TRAINING_TYPE" == "v1.5" ]]; then
    TOTAL_STEPS=200000
    WARMUP_STEPS=20000 #If set, will perform warmup steps with linear decay.
    LR=5e-6
    BATCH_SIZE=128
    ACC_STEPS=1
    NEGS_PER_QUERY=1
    USE_INBATCH_NEGS=""
    IGNORE_PROVIDED_SCORES_IF_ANY="--ignore_scores" #If set, forces CE loss (ColBERTv1) instead of KL-div loss (ColBERTv2) if scores from cross-encoder are provided in the training tuples.
    DISTIL_ALPHA=1.0
fi
if [[ "$TRAINING_TYPE" == "v1.5" ]]; then
    USE_INBATCH_NEGS="--use_ib_negatives"
fi
if [ "$TRAINING_TYPE" = "v2" ]; then
    TOTAL_STEPS=400000
    WARMUP_STEPS=20000
    LR=1e-5
    BATCH_SIZE=32
    ACC_STEPS=4
    NEGS_PER_QUERY=48
    USE_INBATCH_NEGS="--use_ib_negatives"
    IGNORE_PROVIDED_SCORES_IF_ANY=""
    DISTIL_ALPHA=1.0
fi
OUT_DIR="output/training"

# Indexing.
NBITS=2
KMEANS_ITERS=4

# Run.
python src/retrievers/multi_vector_biencoder.py \
    --dataset $DATA \
    --language $LANG \
    $DO_TRAIN \
    --model_name $MODEL \
    --dim $DIM \
    --similarity $SIM \
    --doc_maxlen $DOC_MAXLEN \
    --query_maxlen $QUERY_MAXLEN \
    $MASK_PUNCT_IN_DOCS \
    $ATTEND_MASK_TOKENS_IN_QUERIES \
    --maxsteps $TOTAL_STEPS \
    --warmup $WARMUP_STEPS \
    --lr $LR \
    --bsize $BATCH_SIZE \
    --accumsteps $ACC_STEPS \
    --nway $(($NEGS_PER_QUERY + 1)) \
    $USE_INBATCH_NEGS \
    $IGNORE_PROVIDED_SCORES_IF_ANY \
    --distillation_alpha $DISTIL_ALPHA \
    $DO_TEST \
    --nbits $NBITS \
    --kmeans_niters $KMEANS_ITERS \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR