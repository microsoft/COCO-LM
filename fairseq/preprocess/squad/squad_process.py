# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os, collections
from functools import partial
from multiprocessing import Pool, cpu_count
import pickle
import logging

import numpy as np
import six
from tqdm import tqdm
import argparse
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from fairseq.data.squad.squad_extractor import read_squad_examples, squad_convert_examples_to_features


parser = argparse.ArgumentParser()

parser.add_argument('--sentencepiece-model', type=str)
parser.add_argument('--vocab', type=str)

parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--is-training', action='store_true')
parser.add_argument('--version-2-with-negative', action='store_true')
parser.add_argument('--max-query-length', type=int, default=64)
parser.add_argument('--max-seq-length', type=int, default=384)
parser.add_argument('--doc-stride', type=int, default=128)
args = parser.parse_args()

tokenizer = SentencepieceBPE(args)
examples = read_squad_examples(args.input, args.is_training, args.version_2_with_negative)
features = squad_convert_examples_to_features(examples, tokenizer, args.max_seq_length, args.doc_stride, 
    args.max_query_length, args.is_training)
print('size of examples {}, size of features {}'.format(len(examples), len(features)))
pickle.dump(examples, open(args.output+'_examples.pkl', 'wb'))
pickle.dump(features, open(args.output+'_features.pkl', 'wb'))