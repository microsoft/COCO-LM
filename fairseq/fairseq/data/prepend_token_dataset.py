# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item.new([self.token]), item])
        return item

    @property
    def sizes(self):
        if self.token is not None:
            return np.array(self.dataset.sizes) + 1
        else:
            return self.dataset.sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n
