# COCO-LM (Fairseq)

This directory contains the Fairseq version of scripts for fine-tuning COCO-LM pretrained models on GLUE and SQuAD benchmarks. The scripts are based on the [Fairseq Library](https://github.com/pytorch/fairseq).

Paper: [COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining](https://arxiv.org/abs/2102.08473)

## Requirements

The scripts require Python 3.6+ and Pytorch 1.5.0+. In addition, you need to install the codebase by running:
```
bash install.sh
```

## Pretrained Models

We release two COCO-LM pretrained models, [`cocolm-base`](https://github.com/microsoft/COCO-LM/releases/download/v0.1.0/cocolm-base.tar.gz) and [`cocolm-large`](https://github.com/microsoft/COCO-LM/releases/download/v0.1.0/cocolm-large.tar.gz) (**Note: Please follow the links here to download them; do not use the huggingface version of pretrained models as they are not compatible with Fairseq**), which correspond to the `base++` and `large++` models mentioned in the paper, respectively. You need to extract the models from the archives.

## GLUE Fine-tuning

The [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) benchmark is a collection of sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems. 

**Download GLUE Data**: You can download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory.

**Process GLUE Data**: Since Fairseq training takes binary input files, you need to first preprocess the GLUE data to generate binary files by running the following:
```
cd preprocess/glue
bash process.sh <glue_data_folder> <task_name> <dict_dir> <output>
```
where `<glue_data_folder>` is the path of the raw GLUE data; `<task_name>` is one of the following: `{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`; use `ALL` for preprocessing all the GLUE tasks; `<dict_dir>` is the directory containing two dictionary files `sp.model` and `dict.txt` which can be downloaded [here](https://github.com/microsoft/COCO-LM/releases/download/v0.1.0/dict.tar.gz); `<task_name>` is the output directory for processed GLUE data.

**Fine-Tuning**: After preprocessing the GLUE data, you can run the [`run_glue.sh`](run_glue.sh) script for fine-tuning on each GLUE task. An example for using the script for fine-tuning on MNLI is shown below:
```
ARCH=cocolm_base
TASK=MNLI
PRETRAINED_MODEL_PATH=/path/to/cocolm_base/model.pt
GLUE_DATA_DIR=/path/to/processed/glue_data
OUT_PATH=./glue_finetune/cocolm_base
BSZ=32
LR=2e-5
EPOCH=2
WARMUP=16
SEED=1

export CUDA_VISIBLE_DEVICES=0
bash run_glue.sh $TASK $PRETRAINED_MODEL_PATH $GLUE_DATA_DIR $OUT_PATH $ARCH $EPOCH $WARMUP $BSZ $LR $SEED
```
**Note: The `WARMUP` argument is the reciprocal of the warmup ratio (e.g., `WARMUP=16` means that we are using a 6.25% warmup ratio)**

**Optimal Hyperparameters**: The fine-tuning hyperparameters leading to the best dev set performance in our experiments are shown below (please note that the results and optimal hyperparameters might slightly differ in your runs due to different computation environments):

* COCO-LM base++

|  | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | STS-B |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| BSZ | 32/32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 |
| LR | 2e-5/1e-5 | 3e-5 | 1e-5 | 2e-5 | 2e-5 | 2e-5 | 4e-5 | 4e-5 |
| EPOCH | 2/2 | 5 | 5 | 5 | 10 | 10 | 10 | 10 |
| WARMUP | 16/16 | 16 | 16 | 16 | 16 | 10 | 16 | 16 |
| Result | 90.2/90.0 | 92.2 | 94.2 | 94.6 | 67.3 | 87.4 | 91.2 | 91.8 |

* COCO-LM large++

|  | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | STS-B |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| BSZ | 32/32 | 32 | 32 | 32 | 16 | 32 | 16 | 16 |
| LR | 5e-6/5e-6 | 1e-5 | 7e-6 | 5e-6 | 1e-5 | 2e-5 | 2e-5 | 2e-5 |
| EPOCH | 3/2 | 5 | 2 | 2 | 10 | 5 | 5 | 10 |
| WARMUP | 16/16 | 16 | 16 | 16 | 10 | 16 | 16 | 16 |
| Result | 91.4/91.6 | 92.8 | 95.7 | 96.9 | 73.9 | 91.0 | 92.2 | 92.7 |

## SQuAD 2.0 Fine-tuning 
[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 

**Download & Process SQuAD Data**: Since Fairseq training takes binary input files, you need to first preprocess the SQuAD data to generate pickle files by running the following (SQuAD raw data will be automatically downloaded):
```
cd preprocess/squad
bash process.sh <squad_data_folder> <dict_dir>
```
where `<squad_data_folder>` is the path where raw and processed SQuAD data will be stored to; `<dict_dir>` is the directory containing two dictionary files `sp.model` and `dict.txt` which can be downloaded [here](https://github.com/microsoft/COCO-LM/releases/download/v0.1.0/dict.tar.gz).

**Fine-Tuning**: After preprocessing the SQuAD data, you can run the [`run_squad.sh`](run_squad.sh) script for fine-tuning on SQuAD 2.0. An example for using the script is shown below:
```
ARCH=cocolm_base
PRETRAINED_MODEL_PATH=/path/to/cocolm_base/model.pt
DATA_DIR=/path/to/processed/squad2_data
OUT_PATH=./squad2_finetune/cocolm_base
BSZ=32
LR=2e-5
EPOCH=3
WARMUP=10
SEED=1

export CUDA_VISIBLE_DEVICES=0
bash run_squad.sh $PRETRAINED_MODEL_PATH $DATA_DIR $OUT_PATH $ARCH $EPOCH $WARMUP $BSZ $LR $SEED
```
**Note: The `WARMUP` argument is the reciprocal of the warmup ratio (e.g., `WARMUP=16` means that we are using a 6.25% warmup ratio)**

**Optimal Hyperparameters**: The fine-tuning hyperparameters leading to the best dev set performance in our experiments are shown below (please note that the results and optimal hyperparameters might slightly differ in your runs due to different computation environments):

* COCO-LM base++

|  | EM | F1 |
| ------ | ------ | ------ |
| BSZ | 32 | 32 |
| LR | 2e-5 | 2e-5 |
| EPOCH | 3 | 3 |
| WARMUP | 10 | 10 |
| Result | 85.4 | 88.1 |

* COCO-LM large++

|  | EM | F1 |
| ------ | ------ | ------ |
| BSZ | 32 | 32 |
| LR | 2e-5 | 2e-5 |
| EPOCH | 2 | 3 |
| WARMUP | 16 | 10 |
| Result | 88.3 | 91.0 |
