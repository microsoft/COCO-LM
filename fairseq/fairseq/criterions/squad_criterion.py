# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import math

import os
import torch
import torch.nn.functional as F

from fairseq import metrics, utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.squad import SquadResult, compute_predictions_logits, squad_evaluate


@register_criterion('squad')
class SquadCriterion(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)
        self.head_name = 'question_answering_head'

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--n-best-size', default=20, type=int,
                            help="The number of n-best predictions")
        parser.add_argument('--max-answer-length', default=30, type=int,
                            help="The maximum length of the generated answer")
        parser.add_argument('--version-2-with-negative', action='store_true')

    def forward(self, model, sample, reduce=True):
        features, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=None,
        )
        p_mask = sample['targets']['p_mask']
        if self.training:
            start_positions = sample['targets']['starts']
            end_positions = sample['targets']['ends']
            loss = model.classification_heads[self.head_name].forward(features, start_positions, end_positions, p_mask)
        else:
            loss = torch.zeros(1, dtype=torch.float, device=features.device, requires_grad=True)
            outputs = model.classification_heads[self.head_name].forward(features, p_mask=p_mask)


        sample_size = sample['nsentences']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        if not self.training:
            logging_output['start_logits'] = outputs[0].detach()
            logging_output['end_logits'] = outputs[1].detach()
            logging_output['index'] = sample['id']
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss / sample_size / math.log(2))
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)
        metrics.log_scalar('sample_size', sample_size)

    def context_metrics(self, logging_outputs):
        if self.training:
            return
        all_results = []
        task = self.task
        for log in logging_outputs:
            start_logits = log['start_logits']
            end_logits = log['end_logits']
            indices = log['index']
            for i in range(start_logits.size(0)):
                index = int(indices[i])
                unique_id = task.eval_features[index].unique_id
                result = SquadResult(unique_id, 
                                    start_logits[i].float().cpu().tolist(),
                                    end_logits[i].float().cpu().tolist(),
                                    )
                all_results.append(result)

        output_prediction_file = os.path.join(self.args.save_dir, "predictions.json")
        output_nbest_file = os.path.join(self.args.save_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(self.args.save_dir, "null_odds.json")
        if self.args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.args.save_dir, "null_odds.json")
        else:
            output_null_log_odds_file = None
        predictions, null_scores = compute_predictions_logits(
            task.eval_examples,
            task.eval_features,
            all_results,
            n_best_size=self.args.n_best_size,
            max_answer_length=self.args.max_answer_length,
            do_lower_case=False,
            output_prediction_file=output_prediction_file,
            output_nbest_file=output_nbest_file,
            output_null_log_odds_file=output_null_log_odds_file,
            verbose_logging=False,
            version_2_with_negative=self.args.version_2_with_negative,
            null_score_diff_threshold=0.0,
            tokenizer=task.tokenizer,
        )
        # TODO: implement xlnet's beam search solution
        # predictions = compute_predictions_log_probs(
        #     task.eval_examples,
        #     task.eval_features,
        #     all_results,
        #     n_best_size=self.args.n_best_size,
        #     max_answer_length=self.args.max_answer_length,
        #     output_prediction_file=output_prediction_file,
        #     output_nbest_file=output_nbest_file,
        #     output_null_log_odds_file=output_null_log_odds_file,
        #     start_n_top=self.args.start_n_top,
        #     end_n_top=self.args.end_n_top,
        #     version_2_with_negative=self.args.version_2_with_negative,
        #     tokenizer=task.tokenizer,
        #     verbose_logging=False,
        # )
        eval_result = squad_evaluate(task.eval_examples, predictions, null_scores)
        print(eval_result)
