# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
)

class PoolerLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.dense.bias.data.zero_()

    def forward(
        self, hidden_states, p_mask = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            x.masked_fill_(p_mask, float('-inf'))
        return x



class SQuADHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.start_logits = PoolerLogits(hidden_size)
        self.end_logits = PoolerLogits(hidden_size)

    def forward(
        self,
        hidden_states,
        start_positions=None,
        end_positions=None,
        p_mask = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the first token for the labeled span.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the last token for the labeled span.
            is_impossible (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        end_logits = self.end_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            def loss_fct(logits, targets):
                return F.nll_loss(
                    F.log_softmax(
                        logits.view(-1, logits.size(-1)),
                        dim=-1,
                        dtype=torch.float32,
                    ),
                    targets.view(-1),
                    reduction='sum',
                )
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) * 0.5
            return total_loss
        else:
            return start_logits, end_logits
