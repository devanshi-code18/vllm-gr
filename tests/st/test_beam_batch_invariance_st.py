# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Batch invariance of Beam search under ``VLLM_BATCH_INVARIANT=1``.

A request must return the same ranked SIDs and the same float32 scores, bit
for bit, whether it runs alone or inside batches of other sizes, orders and
prompt lengths, including after a prefix-cache hit. The check runs one engine
per (runner, constraint mode) in a subprocess, because batch invariance and
the ModelRunner are process-wide settings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.paths import REPO_ROOT

BEGIN, END = "<|sid_begin|>", "<|sid_end|>"
WIDTH = 128
STEPS = 3
HISTORY_ITEMS = (0, 4, 20, 52, 120)
# Each prompt runs alone first; every later batch is compared with that result.
BATCHES = ((0, 1), (1, 0), (4, 0), (0, 1, 2, 3), (3, 4, 1, 0), (4, 3, 2, 1, 0, 4, 2, 0))


def _bits(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


def _catalog(rng: random.Random, size: int) -> list[list[str]]:
    sids: set[tuple[int, ...]] = set()
    while len(sids) < size:
        sids.add(tuple(rng.randrange(8192) for _ in range(3)))
    return [[BEGIN, f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>", END] for a, b, c in sorted(sids)]


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    from vllm.inputs import tokens_input

    from vllm_gr.constraints.artifact_builder import build_single_trie_artifact
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    output = Path(args.output)
    rng = random.Random(20261125)
    catalog = _catalog(rng, 20000)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for index, count in enumerate(HISTORY_ITEMS):
        history = ", ".join("".join(rng.choice(catalog)) for _ in range(count))
        text = (f"The user recently watched: {history}. " if history else "") + (
            f"Recommend the next video for user {index}."
        )
        prompts.append(list(tokenizer.encode(text, add_special_tokens=False)))
    table = None
    if args.constrained:
        triples = output / "triples.json"
        triples.write_text(
            json.dumps({"triples": [dict(zip("abc", sid[1:4], strict=True)) for sid in catalog]})
        )
        artifact = output / "constraint_table.safetensors"
        artifact_digest, tokenizer_digest = build_single_trie_artifact(
            Path(args.model), triples, artifact
        )
        table = {
            "path": str(artifact),
            "format": "constraint_table_v1",
            "backend": "cuda",
            "max_top_k": WIDTH,
            "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
        }
    params = BeamSearchParams(
        beam_width=WIDTH,
        max_tokens=STEPS + 2,
        temperature=0.0,
        ignore_eos=True,
        include_stop_str_in_output=True,
        begin_token=BEGIN,
        end_token=END,
        length_penalty=0.0,
    )
    longest = max(len(p) for p in prompts) + 2
    results: dict[str, list[Any]] = {}
    with GRLLM(
        model=args.model,
        trust_remote_code=True,
        gr={
            "beam_graph_enabled": True,
            "worker_decision": True,
            "beam_width": [WIDTH] * STEPS,
            "candidate_width": [WIDTH] * STEPS,
            "max_decode_steps": STEPS,
            "constraint_mode": "table" if table else "none",
            "constraint_table": table,
            "beam_batch_sizes": [1, 2, 4, 8],
        },
        attention_config={"backend": "CUSTOM"},
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=2048,
        max_num_batched_tokens=max(2048, 8 * longest),
        max_num_seqs=WIDTH,
        max_logprobs=WIDTH,
        gpu_memory_utilization=0.6,
    ) as llm:
        calls = 0

        def search(members: tuple[int, ...], shared: bool = False) -> list[list[Any]]:
            nonlocal calls
            calls += 1
            # A unique salt forces a cold Prefill; a shared salt reuses cached blocks.
            requests = [
                tokens_input(
                    list(prompts[m]),
                    cache_salt=f"shared-{m}" if shared else f"call{calls}-row{row}",
                )
                for row, m in enumerate(members)
            ]
            outputs = llm.beam_search(requests, params, output_mode="full")
            return [
                [
                    [list(map(int, s.tokens))[-(STEPS + 2) :], _bits(s.cum_logprob)]
                    for s in o.sequences
                ]
                for o in outputs
            ]

        for warmup in ((0,), (0, 1), (0, 1, 2, 3), tuple(range(5)) + (0, 1, 2)):
            search(warmup)
        alone = {m: search((m,))[0] for m in range(len(prompts))}
        mismatches = []
        cases = [(batch, False) for batch in BATCHES] + [
            ((2,), True),
            ((2,), True),
            ((4, 2, 0, 1), True),
        ]
        for batch, shared in cases:
            for member, beams in zip(batch, search(batch, shared), strict=True):
                if beams != alone[member]:
                    mismatches.append({"batch": batch, "member": member, "cache_hit_case": shared})
                results.setdefault(str(batch), []).append(beams)
    (output / "result.json").write_text(json.dumps({"mismatches": mismatches}, indent=2))
    print(json.dumps({"cases": len(cases), "mismatches": len(mismatches)}))
    return 0 if not mismatches else 1


@pytest.mark.slow
@pytest.mark.model_test
@pytest.mark.l3
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("constrained", [True, False], ids=["table", "full_vocab"])
@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_beam_search_is_batch_invariant(tmp_path: Path, runner: str, constrained: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Beam batch invariance requires CUDA")
    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_BATCH_INVARIANT="1",
        VLLM_USE_V2_MODEL_RUNNER=str(int(runner == "v2")),
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_beam_batch_invariance_st", str(tmp_path), model]
    if constrained:
        command.append("--constrained")
    with subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True) as process:
        try:
            returncode = process.wait(timeout=1400)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    report = tmp_path / "result.json"
    assert returncode == 0, report.read_text() if report.exists() else f"exit {returncode}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("model")
    parser.add_argument("--constrained", action="store_true")
    sys.exit(run(parser.parse_args()))
