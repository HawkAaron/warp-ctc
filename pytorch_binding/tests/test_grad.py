import torch
from warpctc_pytorch import CTCLoss
from torch.autograd import gradcheck

eps = 1e-6
atol = 1e-5
rtol = 1e-3
use_gpu = True

loss = CTCLoss()
def check_grad(B, T, V):
    ilen = torch.randint(T//2, T, [B], dtype=torch.int32)
    olen = torch.randint(1, T//2, [B], dtype=torch.int32)
    label = torch.randint(1, V, [int(olen.sum())], dtype=torch.int32)
    inputs = [torch.randn(T, B, V, dtype=torch.double, requires_grad=True),
            label, ilen, olen]
    if use_gpu:
        inputs[0] = inputs[0].cuda()
    gradcheck(loss, inputs, eps, atol, rtol)
    print(loss(*inputs))

check_grad(11, 29, 17)

