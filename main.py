from kernels import dequantize_nf4, dequantize_nf4_legacy
from peft.utils.integrations import dequantize_module_weight as peft_dequantize
from utils import set_seed, bnb_Linear4bit, unsloth_dequantize
import torch
import triton

major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = major_version >= 8

QUANTILES = [0.5, 0.2, 0.8]
REPS = 500
WARMUP = 25

METHODS = {
    "custom_triton": dequantize_nf4 if HAS_BFLOAT16 else dequantize_nf4_legacy,
    "unsloth": unsloth_dequantize,
    "peft": peft_dequantize,
}

CONFIGS = [
    # (hd,    m,     seed, dtype,            label)
    (2048, 8192, 3407, torch.float16, "2048×8192  fp16"),
    (2048, 8192, 3407, torch.bfloat16, "2048×8192  bf16"),
    (1024, 4096, 3409, torch.bfloat16, "1024×4096  bf16"),
    (4096, 14336, 3408, torch.bfloat16, "4096×14336 bf16"),
]


def benchmark():
    results = {label: {} for *_, label in CONFIGS}

    for hd, m, seed, dt, label in CONFIGS:
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        layer = bnb_Linear4bit(hd, m, dtype=dt).to("cuda")
        layer.weight.quant_state.dtype = dt

        print(f"\n{'=' * 60}")
        print(f"  Config: {label}  ({hd}→{m})")
        print(f"{'=' * 60}")

        for name, fn in METHODS.items():
            ms_50, ms_20, ms_80 = triton.testing.do_bench(
                lambda: fn(layer), warmup=WARMUP, quantiles=QUANTILES, rep=REPS
            )
            results[label][name] = (ms_50, ms_20, ms_80)

    # Calculate median speedup
    for label, methods in results.items():
        unsloth, peft, custom_triton = (
            methods["unsloth"],
            methods["peft"],
            methods["custom_triton"],
        )
        print(f"Config: {label}")
        print(
            f"Custom triton: {custom_triton[0]:.4f} ms, {custom_triton[1]:.4f} ms, {custom_triton[2]:.4f} ms"
        )
        print(f"Peft: {peft[0]:.4f} ms, {peft[1]:.4f} ms, {peft[2]:.4f} ms")
        print(f"Unsloth: {unsloth[0]:.4f} ms, {unsloth[1]:.4f} ms, {unsloth[2]:.4f} ms")
        print(
            f"Speed up: {unsloth[0] / custom_triton[0]:.4f}x vs unsloth, {peft[0] / custom_triton[0]:.4f}x vs peft"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    benchmark()
