import torch
from bitsandbytes.nn import Linear4bit
from unsloth.kernels.utils import fast_dequantize


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bnb_Linear4bit(hd, m, dtype=torch.float16):
    return Linear4bit(
        hd,
        m,
        bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )


def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)
