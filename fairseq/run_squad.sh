#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# Path to pretrained COCO-LM checkpoints
PRETRAINED_MODEL_PATH=$1

# Path to processed SQuAD 2.0 dataset (containing pickle files) 'path/to/squad2_data'
DATA_DIR=$2

# Output path for results and fine-tuned model
OUTPUT_PATH=$3

# Set pretrained model name, from ['cocolm_base', 'cocolm_large']
ARCH=$4

# Set the hyperparameters for the run
N_EPOCH=$5
WARMUP_RATIO=$6
BSZ=$7
LR=$8
SEED=$9

if [ "$ARCH" = "cocolm_base" ]
then
BINS=64
MAX_DIST=128
else
BINS=128
MAX_DIST=256
fi

BETAS="(0.9,0.98)"
CLIP=0.0
WEIGHT_DECAY=0.01

if [ ! -e $PRETRAINED_MODEL_PATH ]; then
    echo "Checkpoint doesn't exist"
    exit 0
fi

EPOCH_ITER=8218
OPTION="--version-2-with-negative"

BSZ_EXPAND=$((BSZ/16))
EPOCH_ITER=$((EPOCH_ITER/BSZ_EXPAND))

TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS/WARMUP_RATIO))
VALIDATE_INTERVAL=$((EPOCH_ITER/2))

OUTPUT_PATH=$OUTPUT_PATH/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'Loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi

python train.py $DATA_DIR --num-workers 0 --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --restore-file $PRETRAINED_MODEL_PATH \
    --max-positions 512 \
    --max-sentences $BSZ \
    --update-freq 1 \
    --task squad \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch $ARCH \
    --criterion squad $OPTION \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "$BETAS" --adam-eps 1e-06 \
    --clip-norm $CLIP \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_STEPS --warmup-updates $WARMUP_STEPS  \
    --max-update $TOTAL_STEPS --seed $SEED --save-dir ./ --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints \
    --find-unused-parameters --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric loss --maximize-best-checkpoint-metric --rel-pos 1 --max-rel-pos $MAX_DIST --rel-pos-bins $BINS \
    --bpe sentencepiece --sentencepiece-model $DATA_DIR/sp.model --vocab $DATA_DIR/dict.txt --validate-interval-updates $VALIDATE_INTERVAL | tee $OUTPUT_PATH/train_log.txt
