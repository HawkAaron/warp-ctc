import torch
import warpctc_pytorch as warp_ctc
from torch.autograd import Function
from torch.nn import Module

from ._warp_ctc import *

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "gradients only computed for acts - please " \
        "mark other tensors as not requiring gradients"

class _CTC(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, size_average=False,
                length_average=False, reduce=True, blank=0):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        loss_func = warp_ctc.gpu_ctc if is_cuda else warp_ctc.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs,
                  blank)

        # costs = torch.FloatTensor([costs.sum()])
        if reduce: costs = costs.sum()

        if length_average:
            # Compute the avg. log-probability per batch sample and frame.
            total_length = torch.sum(act_lens)
            grads /= total_length
            costs /= total_length
        elif size_average:
            # Compute the avg. log-probability per batch sample.
            grads /= minibatch_size
            costs /= minibatch_size

        ctx.grads = grads
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        # same dimension, dtype and device with grads
        scale = grad_output.reshape(1, -1, 1).to(ctx.grads)
        return scale * ctx.grads, None, None, None, None, None, None, None


class CTCLoss(Module):
    """
    Parameters:
        blank (int): blank label
            (default: `0`)
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        length_average (bool): normalize the loss by the total number of frames
            in the batch. If `True`, supersedes `size_average`
            (default: `False`)
        batch_first (bool): batch dim first
            (default: `False`)
        reduce (bool): sum losses
            (default: `True`)
    """
    def __init__(self, blank=0, size_average=False, length_average=False, batch_first=False, reduce=True):
        super(CTCLoss, self).__init__()
        self.ctc = _CTC.apply
        self.blank = blank
        self.size_average = size_average
        self.length_average = length_average
        self.batch_first = batch_first
        self.reduce = reduce

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert len(labels.size()) == 1  # labels must be 1 dimensional
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        if self.batch_first:
            acts = acts.transpose(0, 1)
        return self.ctc(acts, labels, act_lens, label_lens, self.size_average,
                        self.length_average, self.reduce, self.blank)
