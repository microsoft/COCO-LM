# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
## Finetuning COCO-LM for sequence classification on GLUE.
## The script is largely adapted from the huggingface transformers library.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME

from transformers import AdamW, get_linear_schedule_with_warmup
from cocolm.modeling_cocolm import COCOLMForSequenceClassification
from cocolm.configuration_cocolm import COCOLMConfig
from cocolm.tokenization_cocolm import COCOLMTokenizer

from utils_for_glue import glue_compute_metrics as compute_metrics
from utils_for_glue import glue_output_modes as output_modes
from utils_for_glue import glue_processors as processors
from utils_for_glue import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'cocolm': (COCOLMConfig, COCOLMForSequenceClassification, COCOLMTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_optimizer_grouped_parameters(
        model, weight_decay, learning_rate, layer_decay, n_layers, layer_wise_weight_decay=False):
    assert isinstance(model, torch.nn.Module)
    groups = {}
    num_max_layer = 0
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    groups_keys = {}
    for para_name, para_var in model.named_parameters():
        if any(nd in para_name for nd in no_decay):
            weight_decay_in_this_group = 0.0
        else:
            weight_decay_in_this_group = weight_decay
        if para_name.startswith('cocolm.embedding') or para_name == 'cocolm.rel_pos_bias.weight':
            depth = 0
        elif para_name.startswith('cocolm.encoder.layer'):
            depth = int(para_name.split('.')[3]) + 1
            num_max_layer = max(num_max_layer, depth)
        elif para_name.startswith('classifier') or para_name.startswith('cocolm.pooler'):
            depth = n_layers + 2
        else:
            if layer_decay < 1.0:
                logger.warning("para_name %s not find !" % para_name)
                raise NotImplementedError()
            depth = 0

        if layer_decay < 1.0 and layer_wise_weight_decay:
            weight_decay_in_this_group *= (layer_decay ** (n_layers + 2 - depth))
        if layer_decay < 1.0:
            group_name = "layer{}_decay{}".format(depth, weight_decay_in_this_group)
        else:
            group_name = "weight_decay{}".format(weight_decay_in_this_group)
        if group_name not in groups:
            group = {
                "params": [para_var],
                "weight_decay": weight_decay_in_this_group,
            }
            if layer_decay < 1.0:
                group["lr"] = learning_rate * (layer_decay ** (n_layers + 2 - depth))
            groups[group_name] = group
            groups_keys[group_name] = [para_name]
        else:
            group = groups[group_name]
            group["params"].append(para_var)
            groups_keys[group_name].append(para_name)
    print(f"num_max_layer: {num_max_layer}; n_layers: {n_layers}")
    assert num_max_layer == n_layers

    logger.info("Optimizer groups: = %s" % json.dumps(groups_keys))

    return list(groups.values())


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model=model, weight_decay=args.weight_decay, learning_rate=args.learning_rate,
        layer_decay=args.layer_decay, n_layers=model.config.num_hidden_layers,
    )
    
    warmup_steps = t_total * args.warmup_ratio
    correct_bias = not args.disable_bias_correct

    logger.info("*********** Optimizer setting: ***********")
    logger.info("Learning rate = %.10f" % args.learning_rate)
    logger.info("Adam epsilon = %.10f" % args.adam_epsilon)
    logger.info("Adam_betas = (%.4f, %.4f)" % (float(args.adam_betas[0]), float(args.adam_betas[1])))
    logger.info("Correct_bias = %s" % str(correct_bias))
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
        betas=(float(args.adam_betas[0]), float(args.adam_betas[1])),
        correct_bias=correct_bias,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        amp_state_dict = amp.state_dict()
        amp_state_dict['loss_scaler0']['loss_scale'] = args.fp16_init_loss_scale
        logger.info("Set fp16_init_loss_scale to %.1f" % args.fp16_init_loss_scale)
        amp.load_state_dict(amp_state_dict)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    metric_for_best = args.metric_for_choose_best_checkpoint
    best_performance = None
    best_epoch = None
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        if args.disable_tqdm:
            epoch_iterator = train_dataloader
        else:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            
            inputs['token_type_ids'] = None

            if args.model_type in ["cocolm"]:
                longest_input_length = torch.max(inputs["attention_mask"].argmin(dim=1)).item()
                inputs["input_ids"] = inputs["input_ids"][:, :longest_input_length]
                inputs["attention_mask"] = inputs["attention_mask"][:, :longest_input_length]
    
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    if tb_writer is not None:
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{'step': global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                if not args.disable_tqdm:
                    epoch_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            logs = {}
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer, prefix='epoch-{}'.format(_ + 1))
                for key, value in results.items():
                    eval_key = 'eval_{}'.format(key)
                    logs[eval_key] = value

                if metric_for_best is None:
                    metric_for_best = list(list(results.values())[0].keys())[0]
                if best_epoch is None:
                    best_epoch = _ + 1
                    best_performance = results
                else:
                    for eval_task in results:
                        if best_performance[eval_task][metric_for_best] < results[eval_task][metric_for_best]:
                            best_performance[eval_task] = results[eval_task]
                            best_epoch = _ + 1

            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs['learning_rate'] = learning_rate_scalar
            logs['loss'] = loss_scalar
            logging_loss = tr_loss

            if tb_writer is not None:
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
            print(json.dumps({**logs, **{'step': global_step}}))

            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'epoch-{}'.format(_ + 1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not args.do_not_save:
                model_to_save = model.module if hasattr(model, 'module') else model
                # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.fp16:
            logger.info("Amp state dict = %s" % json.dumps(amp.state_dict()))

    if args.local_rank in [-1, 0] and tb_writer is not None:
        tb_writer.close()

    if best_epoch is not None:
        logger.info(" ***************** Best checkpoint: {}, chosen by {} *****************".format(
            best_epoch, metric_for_best))
        logger.info("Best performance = %s" % json.dumps(best_performance))
        save_best_result(best_epoch, best_performance, args.output_dir)

    return global_step, tr_loss / global_step


def save_best_result(best_epoch, best_performance, output_dir):
    best_performance["checkpoint"] = best_epoch
    with open(os.path.join(output_dir, "best_performance.json"), mode="w") as writer:
        writer.write(json.dumps(best_performance, indent=2))


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        cached_dev_file = args.cached_dev_file
        if cached_dev_file is not None:
            cached_dev_file = cached_dev_file + '_' + eval_task
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, cached_features_file=cached_dev_file, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        if args.disable_tqdm:
            epoch_iterator = eval_dataloader
        else:
            epoch_iterator = tqdm(eval_dataloader, desc="Evaluating")
        for batch in epoch_iterator:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                inputs['token_type_ids'] = None

                if args.model_type in ["cocolm"]:
                    longest_input_length = torch.max(inputs["attention_mask"].argmin(dim=1)).item()
                    inputs["input_ids"] = inputs["input_ids"][:, :longest_input_length]
                    inputs["attention_mask"] = inputs["attention_mask"][:, :longest_input_length]

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results[eval_task] = result

        eval_output_dir = os.path.join(eval_output_dir, prefix)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            # for key in sorted(result.keys()):
            #     logger.info("  %s = %s", key, str(result[key]))
            #     writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write(json.dumps(result, indent=2))
            logger.info("Result = %s" % json.dumps(result, indent=2))

    return results


def load_and_cache_examples(args, task, tokenizer, cached_features_file=None, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    if cached_features_file is None:
        if args.disable_auto_cache and args.local_rank != -1:
            logger.warning("Please cache the features in DDP mode !")
            raise RuntimeError()
        if not args.disable_auto_cache:
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
                'dev' if evaluate else 'train',
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str(task)))
    if cached_features_file is not None and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
 
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="unilm", type=str, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_not_save", action='store_true',
                        help="Disable save models after each epoch. ")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    parser.add_argument("--cached_train_file", default=None, type=str,
                        help="Path to cache the train set features. ")
    parser.add_argument("--cached_dev_file", default=None, type=str,
                        help="Path to cache the dev set features. ")
    parser.add_argument('--disable_auto_cache', action='store_true',
                        help='Disable the function for automatic cache the training/dev features.')
    parser.add_argument('--disable_tqdm', action='store_true',
                        help='Disable the tqdm bar. ')

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name_or_path", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--layer_decay", default=1.0, type=float,
                        help="Layer decay rate for the layer-wise learning rate. ")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--adam_betas', '--adam_beta', default='0.9,0.999', type=eval_str_list, metavar='B',
                        help='betas for Adam optimizer')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--disable_bias_correct", action='store_true',
                        help="Disable the bias correction items. ")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")
    parser.add_argument("--dropout_prob", default=None, type=float,
                        help="Set dropout prob, default value is read from config. ")
    parser.add_argument("--cls_dropout_prob", default=None, type=float,
                        help="Set cls layer dropout prob. ")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--metric_for_choose_best_checkpoint', type=str, default=None,
                        help="Set the metric to choose the best checkpoint")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16_init_loss_scale', type=float, default=128.0,
                        help="For fp16: initial value for loss scale.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    if args.local_rank in (-1, 0):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'training_args.json'), mode='w', encoding="utf-8") as writer:
            writer.write(json.dumps(args.__dict__, indent=2, sort_keys=True))

    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.model_type not in ["cocolm"]:
        if not hasattr(config, 'need_pooler') or config.need_pooler is not True:
            setattr(config, 'need_pooler', True)
            
    if args.dropout_prob is not None:
        config.hidden_dropout_prob = args.dropout_prob
        config.attention_probs_dropout_prob = args.dropout_prob

    if args.cls_dropout_prob is not None:
        config.cls_dropout_prob = args.cls_dropout_prob

    logger.info("Final model config for finetuning: ")
    logger.info("%s" % config.to_json_string())

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, 
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, cached_features_file=args.cached_train_file, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)

        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        metric_for_best = args.metric_for_choose_best_checkpoint
        best_performance = None
        best_epoch = None

        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            checkpoint_config = config_class.from_pretrained(checkpoint)
            model = model_class.from_pretrained(checkpoint, config=checkpoint_config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)

            if metric_for_best is None:
                metric_for_best = list(list(result.values())[0].keys())[0]
            if best_epoch is None:
                best_epoch = checkpoint
                best_performance = result
            else:
                for eval_task in result:
                    if best_performance[eval_task][metric_for_best] < result[eval_task][metric_for_best]:
                        best_performance[eval_task] = result[eval_task]
                        best_epoch = checkpoint

        if best_epoch is not None:
            logger.info(" ***************** Best checkpoint: {}, chosen by {} *****************".format(
                best_epoch, metric_for_best))
            logger.info("Best performance = %s" % json.dumps(best_performance))

            save_best_result(best_epoch, best_performance, args.output_dir)


if __name__ == "__main__":
    main()
