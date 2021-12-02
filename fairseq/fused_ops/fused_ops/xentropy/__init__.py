try:
    import torch
    import fused_xentropy_cuda
    from .softmax_xentropy import SoftmaxCrossEntropyLoss
    del torch
    del fused_xentropy_cuda
    del softmax_xentropy
except ImportError as err:
    print("cannot import kernels, please install the package")
