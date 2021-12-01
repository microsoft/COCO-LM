# COCO-LM

This repository contains the scripts for fine-tuning COCO-LM pretrained models on GLUE and SQuAD benchmarks. The scripts are based on the [Huggingface Transformers Library](https://github.com/huggingface/transformers).

Paper: [COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining](https://arxiv.org/abs/2102.08473)

## Requirements

The scripts require Python 3.6+ and the required Python packages can be installed via pip (running in a virtual environment is recommended):
```
pip3 install -r requirements.txt
```

## Pretrained Models

We release two COCO-LM pretrained models, [`cocolm-base`](https://huggingface.co/microsoft/cocolm-base) and [`cocolm-large`](https://huggingface.co/microsoft/cocolm-large), which correspond to the `base++` and `large++` models mentioned in the paper, respectively. You do not need to download them manually as they will be automatically downloaded upon running the training scripts.

## GLUE Fine-tuning

The [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) benchmark is a collection of sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems. 

You can download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory.

You can run the [`run_glue.sh`](run_glue.sh) script for fine-tuning on each GLUE task. An example for using the script for fine-tuning on MNLI is shown below:
```
MODEL=cocolm-base
TASK=MNLI
GLUE_DATASET_PATH=/path/to/downloaded/glue_data
OUT_PATH=./glue_finetune
BSZ=16
LR=1e-5
EPOCH=5
WARMUP=0.0625
SEED=1

export CUDA_VISIBLE_DEVICES=0
bash run_glue.sh $MODEL $TASK $GLUE_DATASET_PATH $OUT_PATH $BSZ $LR $EPOCH $WARMUP $SEED
```

## SQuAD 2.0 Fine-tuning 
[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 

The SQuAD 2.0 dataset will be automatically downloaded upon running the training script.

You can run the [`run_squad.sh`](run_squad.sh) script for fine-tuning on SQuAD 2.0. An example for using the script is shown below:
```
MODEL=cocolm-base
SQUAD_DATASET_PATH=/path/to/squad2_data/
OUT_PATH=./squad_finetune
BSZ=32
LR=3e-5
EPOCH=5
WARMUP=0.0625
SEED=1

export CUDA_VISIBLE_DEVICES=0
bash run_squad.sh $MODEL $SQUAD_DATASET_PATH $OUT_PATH $BSZ $LR $EPOCH $WARMUP $SEED
```

## Citation
If you find the code and models useful for your research, please cite the following paper:
```
@inproceedings{meng2021coco,
  title={{COCO-LM}: Correcting and contrasting text sequences for language model pretraining},
  author={Meng, Yu and Xiong, Chenyan and Bajaj, Payal and Tiwary, Saurabh and Bennett, Paul and Han, Jiawei and Song, Xia},
  booktitle={NeurIPS},
  year={2021}
}
```
