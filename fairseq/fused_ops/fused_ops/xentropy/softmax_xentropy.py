import torch
import fused_xentropy_cuda

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
        if padding_idx >= 0:
            grad_loss.masked_fill_(labels==padding_idx.item(), 0)
        grad_logits = fused_xentropy_cuda.backward(
            grad_loss, logits, max_log_sum_exp,
            labels)

        return grad_logits, None, None, None, None
