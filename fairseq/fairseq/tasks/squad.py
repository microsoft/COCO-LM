# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import pickle
import torch
import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
    RightPadDataset,
    RawLabelDataset,
    RawArrayDataset,
)

#from transformers import BertTokenizer, squad_convert_examples_to_features
#from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

from fairseq.tasks import LegacyFairseqTask, register_task

@register_task('squad')
class SQuADTask(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')

    def __init__(self, args, dictionary):
        super().__init__(args)

        self.dictionary = dictionary
        self.seed = args.seed
        self.tokenizer = encoders.build_bpe(args)
        assert self.tokenizer is not None
        self.dictionary.add_symbol('[MASK]')

    @classmethod
    def load_dictionary(cls, filename):
        dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| Dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        features_file_path = os.path.join(self.args.data, "{}_features.pkl".format(split))
        examples_file_path = os.path.join(self.args.data, "{}_examples.pkl".format(split))

        if os.path.exists(features_file_path) and os.path.exists(examples_file_path):
            examples = pickle.load(open(examples_file_path, 'rb'))
            features = pickle.load(open(features_file_path, 'rb'))
        else:
            raise FileNotFoundError("cannot find {} or {}".format(features_file_path, examples_file_path))

        if split == 'valid':
            # save for eval
            self.eval_examples = examples
            self.eval_features = features

        src_tokens = RawArrayDataset([torch.from_numpy(np.array(f.input_ids)) for f in features])
        p_mask = RawArrayDataset([torch.from_numpy(np.array(f.p_mask)).bool() for f in features])
        if split == 'train':
            starts = RawLabelDataset([int(f.start_position) for f in features])
            ends = RawLabelDataset([int(f.end_position) for f in features])
            is_impossible = RawLabelDataset([int(f.is_impossible) for f in features])
        else:
            starts = ends = is_impossible = None
        #sizes = np.array([len(f.input_ids) for f in features])

        '''
            Input format: <s> question here ? </s> Passage </s>
        '''
        dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': RightPadDataset(
                        src_tokens,
                        pad_idx=self.dictionary.pad(),
                    ),  
                    'src_lengths': NumelDataset(src_tokens, reduce=False),
                },
                'targets': {
                    'starts': starts,
                    'ends': ends,
                    'is_impossible': is_impossible,
                    'p_mask': RightPadDataset(p_mask, pad_idx=1),
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_tokens, reduce=True),
            },
            sizes=[src_tokens.sizes],
        )

        if split == 'train':
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_tokens))
            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_question_answering_head(
            'question_answering_head',
            num_classes=2,
        )
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

