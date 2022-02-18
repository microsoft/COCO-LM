# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# The script is largely adapted from the huggingface transformers library

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

COCOLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'microsoft/cocolm-base': "https://huggingface.co/microsoft/cocolm-base/resolve/main/config.json",
    'microsoft/cocolm-large': "https://huggingface.co/microsoft/cocolm-large/resolve/main/config.json",
}

class COCOLMConfig(PretrainedConfig):
    model_type = "cocolm"
    pretrained_config_archive_map = COCOLM_PRETRAINED_CONFIG_ARCHIVE_MAP
    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        summary_type="first",
        summary_use_proj=True,
        summary_activation="gelu",
        summary_last_dropout=0.1,
        pad_token_id=0,
        rel_pos_bins=0,
        max_rel_pos=0,
        layer_norm_type='post',
        **kwargs
    ):
        super(COCOLMConfig, self).__init__(**kwargs)
        if isinstance(vocab_size, str) or (sys.version_info[0] == 2
                                           and isinstance(vocab_size, unicode)):
            with open(vocab_size, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps

            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_last_dropout = summary_last_dropout
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.layer_norm_type = layer_norm_type
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
