# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

DATASET_PATH=$1
DICT_PATH=$2

mkdir -p $DATASET_PATH

cp $DICT_PATH/sp.model $DATASET_PATH
cp $DICT_PATH/dict.txt $DATASET_PATH

export TRAIN_FILE=$DATASET_PATH/train-v2.0.json
if [ ! -f $TRAIN_FILE ]
then
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $TRAIN_FILE
fi

python squad_process.py --input $TRAIN_FILE --output $DATASET_PATH/train \
                        --sentencepiece-model $DICT_PATH/sp.model --vocab $DICT_PATH/dict.txt \
                        --is-training --version-2-with-negative

export DEV_FILE=$DATASET_PATH/dev-v2.0.json
if [ ! -f $DEV_FILE ]
then
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $DEV_FILE
fi

python squad_process.py --input $DEV_FILE --output $DATASET_PATH/valid \
                        --sentencepiece-model $DICT_PATH/sp.model --vocab $DICT_PATH/dict.txt \
                        --version-2-with-negative
