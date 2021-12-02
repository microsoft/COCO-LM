# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# Set pretrained model name, from ['cocolm-base', 'cocolm-large']
MODEL_NAME=$1

# Path to SQuAD dataset 'path/to/squad2_data'
DATASET_PATH=$2

# Output path for results and fine-tuned model
OUT_PATH=$3

mkdir -p $DATASET_PATH
# Train datset
export TRAIN_FILE=$DATASET_PATH/train-v2.0.json
if [ ! -f $TRAIN_FILE ]
then
	wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $TRAIN_FILE
fi
# Dev datset
export DEV_FILE=$DATASET_PATH/dev-v2.0.json
if [ ! -f $DEV_FILE ]
then
	wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $DEV_FILE
fi

# Set max sequence length
export MAX_LEN=384

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${TRAIN_FILE}_cocolm_cased.384doc_new.cache
export DEV_CACHE=${DEV_FILE}_cocolm_cased.384doc_new.cache

# Setting the hyperparameters for the run.
export BSZ=$4
export LR=$5
export EPOCH=$6
export WM=$7
export SEED=$8

# Set path to save the finetuned model and result score
export OUTPUT_PATH=$OUT_PATH/$BSZ-$LR-$EPOCH-$WM-$SEED

mkdir -p $OUTPUT_PATH
touch $OUTPUT_PATH/train.log

python run_squad.py \
    --model_type cocolm --model_name_or_path $MODEL_NAME \
    --config_name $MODEL_NAME --tokenizer_name_or_path cocolm-cased \
    --train_file $TRAIN_FILE --predict_file $DEV_FILE \
    --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --do_train --evaluate_during_training --logging_steps 1000 \
    --per_gpu_train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH --gradient_accumulation_steps 1 \
    --max_seq_length $MAX_LEN --doc_stride 128 --output_dir $OUTPUT_PATH \
    --version_2_with_negative --seed 1 --max_grad_norm 0 \
    --weight_decay 0.01 --warmup_ratio $WM  \
    --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --seed $SEED \
    --overwrite_output_dir \
    --metric_for_choose_best_checkpoint "best_f1" |& tee $OUTPUT_PATH/train.log

# Add the following for fp16 training    
# --fp16_init_loss_scale 128.0 --fp16 --fp16_opt_level O2
