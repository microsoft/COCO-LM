# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _cross_entropy_pytorch(logits, target, ignore_index=None, reduction="mean"):
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    return F.nll_loss(
        lprobs,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )


try:
    import fused_xentropy_cuda

    logger.info("using fused cross entropy")
    class SoftmaxCrossEntropyLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels, padding_idx=0, half_to_float=False):
            losses, max_log_sum_exp = fused_xentropy_cuda.forward(
                logits, labels, half_to_float)
            if padding_idx >= 0:
                losses.masked_fill_(labels==padding_idx, 0)
            ctx.save_for_backward(logits, max_log_sum_exp, labels,
                torch.LongTensor([padding_idx]))

            return losses

        @staticmethod
        def backward(ctx, grad_loss):
            logits, max_log_sum_exp, labels, padding_idx = ctx.saved_tensors
            if not grad_loss.is_contiguous():
                grad_loss = grad_loss.contiguous()
            padding_idx = padding_idx.item()
            if padding_idx >= 0:
                grad_loss.masked_fill_(labels==padding_idx, 0)
            grad_logits = fused_xentropy_cuda.backward(
                grad_loss.contiguous(), logits, max_log_sum_exp,
                labels)

            return grad_logits, None, None, None

    def cross_entropy(logits, target, ignore_index=-100, reduction="mean"):
        if logits.device == torch.device("cpu"):
            return _cross_entropy_pytorch(logits, target, ignore_index, reduction)
        else:
            half_to_float = (logits.dtype == torch.half) or (logits.dtype == torch.bfloat16)
            losses = SoftmaxCrossEntropyLoss.apply(
                logits, target, ignore_index, half_to_float,
            )
            if reduction == "sum":
                return losses.sum()
            elif reduction == "mean":
                if ignore_index >= 0:
                    return losses.sum() / target.ne(ignore_index).sum()
                else:
                    return losses.mean()
            elif reduction == "none":
                return losses
            else:
                raise NotImplementedError


except ImportError:

    def cross_entropy(logits, target, ignore_index=-100, reduction="mean"):
        return _cross_entropy_pytorch(logits, target, ignore_index, reduction)
