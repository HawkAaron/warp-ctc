# PyTorch bindings for Warp-ctc

[![Build Status](https://travis-ci.org/SeanNaren/warp-ctc.svg?branch=pytorch_bindings)](https://travis-ci.org/SeanNaren/warp-ctc)

This is an extension onto the original repo found [here](https://github.com/baidu-research/warp-ctc).

## CPU Performance
Benchmarked on a dual-socket machine with two Intel E5-2660 v4 processors - warp-ctc used 10 threads to maximally take advantage of the CPU resources. 

| **T=150, L=40, A=28** | **warp-ctc** | 
| --------------------- | ------------ |
|         N=1           |   1.89 ms    |
|         N=16          |   4.40 ms    |
|         N=32          |   6.39 ms    |
|         N=64          |   10.77 ms   |
|         N=128         |   19.69 ms   |

| **T=150, L=20, A=5000** | **warp-ctc** |
| ----------------------- | ------------ |
|         N=1             |   10.22 ms   |
|         N=16            |   23.26 ms   |
|         N=32            |   44.70 ms   |
|         N=64            |   79.29 ms   |
|         N=128           |   146.83 ms  |

## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation).

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_CTC_PATH` to wherever you have `libwarpctc.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live). For example:

```
export CUDA_HOME="/usr/local/cuda"
```

Now install the bindings:
```
cd pytorch_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 (as recommended by pytorch):
```
cd ../pytorch_binding
python setup.py install
cd ../build
cp libwarpctc.dylib /Users/$WHOAMI/anaconda3/lib
```
This will resolve the library not loaded error. This can be easily modified to work with other python installs if needed.

Example to use the bindings below.

```python
    import torch
    from torch.autograd import Variable
    from warpctc_pytorch import CTCLoss
    ctc_loss = CTCLoss()
    # expected shape of seqLength x batchSize x alphabet_size
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    labels = Variable(torch.IntTensor([1, 2]))
    label_sizes = Variable(torch.IntTensor([2]))
    probs_sizes = Variable(torch.IntTensor([2]))
    probs = Variable(probs, requires_grad=True) # tells autograd to compute gradients for probs
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()
```

## Documentation

```
CTCLoss(size_average=False, length_average=False)
    # size_average (bool): normalize the loss by the batch size (default: False)
    # length_average (bool): normalize the loss by the total number of frames in the batch. If True, supersedes size_average (default: False)

forward(acts, labels, act_lens, label_lens)
    # acts: Tensor of (seqLength x batch x outputDim) containing output activations from network (before softmax)
    # labels: 1 dimensional Tensor containing all the targets of the batch in one large sequence
    # act_lens: Tensor of size (batch) containing size of each output sequence from the network
    # label_lens: Tensor of (batch) containing label length of each example
```