# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-any-return"
"""The fused Decode binding writes exactly what the eager composition writes."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.cuda.decode_binding import bind_decode_inputs_cuda
from vllm_gr.v1.worker.cuda.gr_inputs import (
    ERROR_DISPATCH_STATE,
    bind_device_inputs_eager,
)

pytestmark = [pytest.mark.gpu_test, pytest.mark.l2]

BATCH = 8
BEAMS = 1024
STEPS = 3


def _owner(rows: list[int], width: int, prefix_width: int, step: int, seed: int) -> SimpleNamespace:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def ints(*shape: int, low: int = 0, high: int = 1 << 20) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=g, device="cuda", dtype=torch.int32)

    def bools(*shape: int, p: float = 0.8) -> torch.Tensor:
        return torch.rand(shape, generator=g, device="cuda") < p

    # Mostly consistent rows so that both running and failing requests occur.
    good = bools(BATCH, p=0.7)
    state = SimpleNamespace(
        is_initialized=bools(BATCH, p=0.9),
        is_finished=bools(BATCH, p=0.15),
        error_code=torch.where(bools(BATCH, p=0.85), 0, ints(BATCH, low=1, high=6)),
        num_generated_tokens=torch.where(good, step + 1, ints(BATCH, high=4)),
        num_decode_steps=torch.where(good, step, ints(BATCH, high=4)),
        active_mask=bools(BATCH, BEAMS),
        input_ids=ints(BATCH, BEAMS),
        sequence=ints(BATCH, BEAMS + 3, STEPS)[:, :BEAMS],
    )
    n = len(rows) * width
    metadata = SimpleNamespace(
        suffix_seqlens_kv_device=torch.full((n,), -7, device="cuda", dtype=torch.int32),
        suffix_cu_seqlens_k_device=torch.full((n + 1,), -7, device="cuda", dtype=torch.int32),
    )
    context = SimpleNamespace(
        decode_step_buf=torch.full((1,), -7, device="cuda", dtype=torch.int32)
    )
    return SimpleNamespace(
        beam=SimpleNamespace(state=state, session=SimpleNamespace(context=context)),
        state_rows=torch.tensor(rows, device="cuda", dtype=torch.long),
        device_positions=torch.arange(n, device="cuda", dtype=torch.int64) * 3 + 11,
        execution_mask=torch.zeros(n, device="cuda", dtype=torch.bool),
        input_ids=torch.full((n,), -7, device="cuda", dtype=torch.int32),
        positions=torch.full((n,), -7, device="cuda", dtype=torch.int64),
        prefixes=torch.full((n, prefix_width), -7, device="cuda", dtype=torch.int32),
        metadata=metadata,
    )


def _outputs(owner: SimpleNamespace) -> dict[str, torch.Tensor]:
    return {
        "error_code": owner.beam.state.error_code,
        "execution_mask": owner.execution_mask,
        "input_ids": owner.input_ids,
        "positions": owner.positions,
        "decode_step": owner.beam.session.context.decode_step_buf,
        "prefixes": owner.prefixes,
        "suffix_lengths": owner.metadata.suffix_seqlens_kv_device,
        "suffix_cu": owner.metadata.suffix_cu_seqlens_k_device,
    }


def _fused(owner: SimpleNamespace, width: int, step: int) -> bool:
    state = owner.beam.state
    return bind_decode_inputs_cuda(
        state_rows=owner.state_rows,
        is_initialized=state.is_initialized,
        is_finished=state.is_finished,
        error_code=state.error_code,
        num_generated_tokens=state.num_generated_tokens,
        num_decode_steps=state.num_decode_steps,
        active_mask=state.active_mask,
        state_input_ids=state.input_ids,
        sequence=state.sequence,
        device_positions=owner.device_positions,
        out_execution_mask=owner.execution_mask,
        out_input_ids=owner.input_ids,
        out_positions=owner.positions,
        out_decode_step=owner.beam.session.context.decode_step_buf,
        out_prefixes=owner.prefixes,
        out_suffix_lengths=owner.metadata.suffix_seqlens_kv_device,
        out_suffix_cu=owner.metadata.suffix_cu_seqlens_k_device,
        width=width,
        step=step,
        dispatch_error=ERROR_DISPATCH_STATE,
    )


CASES = [
    (rows, width, step, prefix_width, seed)
    for (rows, width), step, prefix_width, seed in itertools.product(
        [
            ([0], 1),
            ([0], 128),
            ([5], 256),
            ([0, 1], 128),
            ([6, 2], 64),
            ([3, 1, 7, 0], 32),
            ([0, 1, 2, 3], 256),
            (list(range(8)), 128),
            ([7, 6, 5, 4, 3, 2, 1, 0], 1024),
        ],
        [0, 1, 2],
        [3, 4],
        [0, 1],
    )
]


@pytest.mark.parametrize(("rows", "width", "step", "prefix_width", "seed"), CASES)
def test_fused_binding_matches_eager(
    rows: list[int], width: int, step: int, prefix_width: int, seed: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    expected = _owner(rows, width, prefix_width, step, seed)
    actual = _owner(rows, width, prefix_width, step, seed)
    bind_device_inputs_eager(expected, step, width)
    assert _fused(actual, width, step)
    for name, tensor in _outputs(expected).items():
        assert torch.equal(_outputs(actual)[name], tensor), name


def test_unsupported_layout_falls_back() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    owner = _owner([0, 1], 16, 3, 1, 0)
    owner.input_ids = torch.empty(64, device="cuda", dtype=torch.int32)[::2]
    assert not _fused(owner, 16, 1)
