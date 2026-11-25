# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One-launch CUDA Decode input binding from persistent Beam state.

Replaces about two dozen small eager PyTorch operations per Decode dispatch
(row gathers, masks, ``where``, ``fill``, ``repeat_interleave``, ``cumsum``)
with one Triton program. The values are integers and booleans, so the result
is identical to the eager composition.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit(do_not_specialize=["step", "columns"])
def _bind_decode_inputs_kernel(
    state_rows,  # int64 [R]: Beam state row of each logical request
    is_initialized,  # bool [batch]
    is_finished,  # bool [batch]
    error_code,  # int32 [batch], updated in place
    num_generated_tokens,  # int32 [batch]
    num_decode_steps,  # int32 [batch]
    active_mask,  # bool [batch, beam_width] (row stride below)
    state_input_ids,  # int32 [batch, beam_width]
    sequence,  # int32 [batch, beam_width, steps]
    device_positions,  # int64 [R * W]
    out_execution_mask,  # bool [R * W]
    out_input_ids,  # [R * W]
    out_positions,  # [R * W]
    out_decode_step,  # int32 [1]
    out_prefixes,  # int32 [R * W, P]
    out_suffix_lengths,  # [R * W]
    out_suffix_cu,  # [R * W + 1]
    R,
    W,
    step,
    columns,
    active_stride,
    ids_stride,
    seq_stride_b,
    seq_stride_w,
    prefix_stride,
    dispatch_error,
    P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Flat Beam rows. Every lane recomputes its request's dispatch check from
    # the loaded state, so no lane depends on the error-code store below.
    i = tl.arange(0, BLOCK_N)
    n = R * W
    lane = i < n
    req = i // W
    beam = i - req * W
    lane_row = tl.load(state_rows + req, mask=lane, other=0)
    l_init = tl.load(is_initialized + lane_row, mask=lane, other=0) != 0
    l_fin = tl.load(is_finished + lane_row, mask=lane, other=0) != 0
    l_err = tl.load(error_code + lane_row, mask=lane, other=0)
    l_count = tl.load(num_generated_tokens + lane_row, mask=lane, other=0)
    l_actual = tl.load(num_decode_steps + lane_row, mask=lane, other=0)
    l_mismatch = (~l_fin) & (l_err == 0) & ((~l_init) | (l_actual != step) | (l_count != step + 1))
    l_err = tl.where(l_mismatch, dispatch_error, l_err)
    l_running = l_init & (~l_fin) & (l_err == 0) & lane
    active = tl.load(active_mask + lane_row * active_stride + beam, mask=lane, other=0) != 0
    mask = l_running & active
    tl.store(out_execution_mask + i, mask, mask=lane)
    ids = tl.load(state_input_ids + lane_row * ids_stride + beam, mask=lane & mask, other=0)
    tl.store(out_input_ids + i, tl.where(mask, ids, 0), mask=lane)
    pos = tl.load(device_positions + i, mask=lane & mask, other=0)
    tl.store(out_positions + i, tl.where(mask, pos, 0), mask=lane)

    # Request rows: the same check, published once per row after all lane loads.
    r = tl.arange(0, BLOCK_R)
    r_valid = r < R
    row = tl.load(state_rows + r, mask=r_valid, other=0)
    initialized = tl.load(is_initialized + row, mask=r_valid, other=0) != 0
    finished = tl.load(is_finished + row, mask=r_valid, other=0) != 0
    error = tl.load(error_code + row, mask=r_valid, other=0)
    count = tl.load(num_generated_tokens + row, mask=r_valid, other=0)
    actual = tl.load(num_decode_steps + row, mask=r_valid, other=0)
    mismatch = (
        (~finished) & (error == 0) & ((~initialized) | (actual != step) | (count != step + 1))
    )
    error = tl.where(mismatch, dispatch_error, error)
    running = initialized & (~finished) & (error == 0)
    all_running = tl.sum(tl.where(r_valid & (~running), 1, 0), axis=0) == 0
    tl.store(out_decode_step, tl.where(all_running, step, -1))
    tl.debug_barrier()
    tl.store(error_code + row, error, mask=r_valid)

    c = tl.arange(0, BLOCK_P)
    c_valid = c < P
    tokens = tl.load(
        sequence + lane_row[:, None] * seq_stride_b + beam[:, None] * seq_stride_w + c[None, :],
        mask=(lane & mask)[:, None] & (c[None, :] < columns),
        other=-1,
    )
    prefix = tl.where(mask[:, None] & (c[None, :] < columns), tokens, -1)
    tl.store(
        out_prefixes + i[:, None] * prefix_stride + c[None, :],
        prefix,
        mask=lane[:, None] & c_valid[None, :],
    )

    lengths = tl.where(l_running, step + 1, 1)
    lengths = tl.where(lane, lengths, 0)
    tl.store(out_suffix_lengths + i, lengths, mask=lane)
    tl.store(out_suffix_cu + 1 + i, tl.cumsum(lengths, axis=0), mask=lane)
    tl.store(out_suffix_cu + i, tl.zeros_like(lengths), mask=i == 0)


def bind_decode_inputs_cuda(
    *,
    state_rows: torch.Tensor,
    is_initialized: torch.Tensor,
    is_finished: torch.Tensor,
    error_code: torch.Tensor,
    num_generated_tokens: torch.Tensor,
    num_decode_steps: torch.Tensor,
    active_mask: torch.Tensor,
    state_input_ids: torch.Tensor,
    sequence: torch.Tensor,
    device_positions: torch.Tensor,
    out_execution_mask: torch.Tensor,
    out_input_ids: torch.Tensor,
    out_positions: torch.Tensor,
    out_decode_step: torch.Tensor,
    out_prefixes: torch.Tensor,
    out_suffix_lengths: torch.Tensor,
    out_suffix_cu: torch.Tensor,
    width: int,
    step: int,
    dispatch_error: int,
) -> bool:
    """Bind one Decode dispatch in one launch; return False to use the eager path."""

    rows = int(state_rows.numel())
    n = rows * width
    columns = min(step + 1, out_prefixes.shape[1], sequence.shape[2])
    flat_outputs = (out_execution_mask, out_input_ids, out_positions, out_suffix_lengths)
    if (
        rows == 0
        or width <= 0
        or not state_rows.is_cuda
        or state_rows.dtype != torch.int64
        or any(t.numel() != n or not t.is_contiguous() for t in flat_outputs)
        or device_positions.numel() != n
        or not device_positions.is_contiguous()
        or out_suffix_cu.numel() != n + 1
        or not out_suffix_cu.is_contiguous()
        or out_prefixes.dim() != 2
        or out_prefixes.shape[0] != n
        or out_prefixes.stride(1) != 1
        or out_decode_step.numel() != 1
        or active_mask.stride(1) != 1
        or state_input_ids.stride(1) != 1
        or sequence.stride(2) != 1
        or any(
            not t.is_contiguous()
            for t in (
                is_initialized,
                is_finished,
                error_code,
                num_generated_tokens,
                num_decode_steps,
            )
        )
    ):
        return False
    prefix_width = int(out_prefixes.shape[1])
    _bind_decode_inputs_kernel[(1,)](
        state_rows,
        is_initialized,
        is_finished,
        error_code,
        num_generated_tokens,
        num_decode_steps,
        active_mask,
        state_input_ids,
        sequence,
        device_positions,
        out_execution_mask,
        out_input_ids,
        out_positions,
        out_decode_step,
        out_prefixes,
        out_suffix_lengths,
        out_suffix_cu,
        rows,
        width,
        step,
        columns,
        active_mask.stride(0),
        state_input_ids.stride(0),
        sequence.stride(0),
        sequence.stride(1),
        out_prefixes.stride(0),
        dispatch_error,
        P=prefix_width,
        BLOCK_R=triton.next_power_of_2(rows),
        BLOCK_N=triton.next_power_of_2(n),
        BLOCK_P=triton.next_power_of_2(prefix_width),
        num_warps=8 if n >= 2048 else 4,
    )
    return True
