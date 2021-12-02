try:
    import torch
    import fused_layernorm_cuda
    from .fused_layer_norm import FusedLayerNorm
    del torch
    del fused_layernorm_cuda
    del fused_layer_norm
except ImportError as err:
    print("cannot import kernels, please install the package")
