# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

from . import BaseWrapperDataset, LRUCacheDataset

class SpanDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for sampling contigous span
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        seed: int = 1,
        span: float = 0,
    ):
        self.dataset = LRUCacheDataset(dataset)
        self.span = span
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)
            if self.span > 1:
                span_length = min(int(self.span), sz)
            else:
                span_length = int(self.span * sz)
            start_idx = np.random.randint(0, sz - span_length + 1)
            new_item = item.clone()
            # it seems [CLS] could be removed in span, is that expected?
            return new_item[start_idx: start_idx + span_length]