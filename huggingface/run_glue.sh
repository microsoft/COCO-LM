# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# Set pretrained model name, from ['cocolm-base', 'cocolm-large']
MODEL_NAME=$1

# GLUE task name, from ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'RTE', 'MRPC', 'STS-B']
TASK=$2

# Path to GLUE dataset 'path/to/glue_data'
GLUE_PATH=$3

# Output path for results and fine-tuned model
OUT_PATH=$4

export DATASET_PATH=$GLUE_PATH/$TASK

export TASK_NAME=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

# Set max sequence length
export MAX_LEN=512

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.cocolm_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.cocolm_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run.
export BSZ=$5
export LR=$6
export EPOCH=$7
export WM=$8
export SEED=$9

# Set path to save the finetuned model and result score
export OUTPUT_PATH=$OUT_PATH/$TASK-$BSZ-$LR-$EPOCH-$WM-$SEED

mkdir -p $OUTPUT_PATH
touch $OUTPUT_PATH/train.log

python run_glue.py \
    --model_type cocolm --model_name_or_path $MODEL_NAME --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $MODEL_NAME --tokenizer_name_or_path cocolm-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 --gradient_accumulation_steps 1 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay 0.01 --warmup_ratio $WM \
    --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --seed $SEED \
    --overwrite_output_dir |& tee $OUTPUT_PATH/train.log

# Add the following for fp16 training
# --fp16_init_loss_scale 128.0 --fp16 --fp16_opt_level O2 
