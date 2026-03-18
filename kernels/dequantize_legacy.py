import triton
import triton.language as tl
import torch


@triton.jit
def _dequantize_nf4_kernel_legacy(
    # Inputs
    weight_ptr,
    absmax_ptr,
    absmax2_ptr,
    code_ptr,
    code2_ptr,
    offset_ptr,
    # Output
    out_ptr,
    # Params
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Block and element offsets
    block_id = tl.program_id(0)
    weight_offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs = block_id * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    out_mask = out_offs < (num_elements * 2)
    weight_mask = weight_offs < num_elements

    # Quantization hierarchy offsets
    absmax_offs = out_offs // 64
    absmax2_offs = absmax_offs // 256

    # Load quantized weights and unpack two NF4 values per byte
    weight = tl.load(weight_ptr + weight_offs, mask=weight_mask, other=0).to(tl.uint8)
    nf4_indices = tl.interleave(weight >> 4, weight & 0x0F)

    # Dequantize: NF4 index -> float via codebook lookup
    nf4_vals = tl.load(code_ptr + nf4_indices, eviction_policy="evict_last")

    # Dequantize absmax: uint8 index -> float via second codebook lookup
    absmax_indices = tl.load(absmax_ptr + absmax_offs, mask=out_mask, other=0).to(
        tl.uint8
    )
    absmax_vals = tl.load(code2_ptr + absmax_indices, eviction_policy="evict_last")
    absmax2_vals = tl.load(absmax2_ptr + absmax2_offs, mask=out_mask, other=1.0)

    # Reconstruct final absmax scale: absmax * absmax2 + offset
    offset = tl.load(offset_ptr, eviction_policy="evict_last")
    offset = tl.broadcast_to(offset, absmax_vals.shape)
    absmax_full = absmax_vals * absmax2_vals
    absmax_scaled = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        args=(absmax_full, offset),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

    # Scale NF4 values and store
    out = (nf4_vals * absmax_scaled).to(out_ptr.type.element_ty)
    if out_ptr.type.element_ty == tl.bfloat16:
        out = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 %r<4>;
                .reg .b16 %h<2>;
                .reg .pred %p<3>;

                // Load input into register
                mov.u32         %r0, $1;

                // Step 1: Extract upper 16 bits
                shr.u32         %r1, %r0, 16;

                // Step 2: Compute rounding (0x7FFF + LSB trick)
                and.b32         %r2, %r1, 0x1;        // Get LSB
                add.u32         %r2, %r2, 0x7FFF;     // Bias = 0x7FFF + LSB
                add.u32         %r3, %r0, %r2;        // Add bias
                shr.u32         %r3, %r3, 16;         // Shift result

                // Step 3: Check for NaN/Inf
                and.b32         %r0, %r0, 0x7F800000; // Mask exponent
                setp.eq.u32     %p0, %r0, 0x7F800000; // Is special?

                // Step 4: Preserve NaN (set mantissa bit if needed)
                and.b32         %r1, %r3, 0x7F;       // Get bf16 mantissa
                setp.eq.u32     %p1, %r1, 0;          // Is it zero?
                and.pred        %p2, %p0, %p1;        // Special AND no mantissa
                @%p2 or.b32     %r3, %r3, 0x40;       // Set mantissa bit

                // Step 5: Convert to 16-bit output
                cvt.u16.u32     %h0, %r3;
                mov.u16         $0, %h0;
            }
            """,
            constraints="=h,r",  # output: 16-bit, input: 32-bit
            dtype=tl.bfloat16,
            args=[out],
            pack=1,
            is_pure=True,
        )
    else:
        out = out.to(tl.float16)
    tl.store(out_ptr + out_offs, out, mask=out_mask)


def _dequantize_nf4_legacy(weight, quant_state):
    num_elements = weight.numel()
    out = torch.empty(2 * num_elements, dtype=quant_state.dtype, device=weight.device)
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)  # noqa
    _dequantize_nf4_kernel_legacy[grid](
        weight,
        out,
        quant_state.absmax,
        quant_state.state2.absmax,
        quant_state.code,
        quant_state.state2.code,
        quant_state.offset,
        num_elements,
        BLOCK_SIZE=256,
    )
    return out.view(quant_state.shape)


def dequantize_nf4_legacy(weight):
    return _dequantize_nf4_legacy(weight.weight.data, weight.weight.quant_state)
