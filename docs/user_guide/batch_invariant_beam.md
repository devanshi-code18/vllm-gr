# Batch-invariant Beam search on CUDA

With vLLM's batch-invariant mode, a CUDA Beam request returns the same ranked
SIDs and the same `cumulative_logprob` values, bit for bit, whether it runs
alone or batched with other requests. The batch can have any size, order or
mix of prompt lengths. Use it when results must be reproducible across load,
for example for offline evaluation, A/B parity checks or replay debugging.

## Enable it

Set the variable before the engine starts. It applies to the whole process.

```bash
export VLLM_BATCH_INVARIANT=1
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

## What stays identical

The [batch-invariance system test](../../tests/st/test_beam_batch_invariance_st.py)
compares each prompt alone with the same prompt batched at sizes 2, 4 and 8,
in different orders and with prompts of different lengths. It also covers
prefix-cache hits. It runs ModelRunner V1 and V2, with and without a
constraint table, using Beam and Prefill graphs.

- Batch invariance holds for every returned beam: token IDs, rank and the
  float32 score.
- Results are not identical across execution modes. ModelRunner V1 and V2
  return the same bits. Eager execution (`enforce_eager=True`) returns
  different bits from compiled execution, because torch.compile fuses
  different kernels.
- Without `VLLM_BATCH_INVARIANT`, results change with batch composition,
  including the top-ranked SID. Running the same batch again still gives the
  same result.

## What it disables

Batch-invariant mode replaces the dense matmul, softmax, mean and RMSNorm
kernels with vLLM's batch-invariant implementations. vLLM-GR also turns off
the CUDA paths whose choice depends on the batch:

| Path | Batch-dependent choice it would make |
| --- | --- |
| Prefix attention splits and [segmentation](../design/feature/cuda_decode_attention.md) | Split count by prefix length and batch |
| Dense Prefill | cuDNN or FlashAttention by request count |
| Vocabulary-sliced LM head (full vocabulary) | Slicing only when a step has at least `head_slice_min_rows` rows |
| Independent Prefill FlashAttention version | FA3 for Prefill and FA2 for Decode |

## Cost

The cost depends on the device. vLLM 0.30.0 ships tuned batch-invariant
matmul tables for SM89, SM90 and SM100, but not for SM80 (A100). On A100 the
default tiles make the decode-sized GEMMs of OneRec-1.7B 1.1x to 3.9x slower
than cuBLAS.
Measure your own workload with the
[Beam performance workload](../cli/bench/beam_performance.md), running it once
with the variable set and once without.

> Measured on A100-SXM4-80GB / 2026-11-25 / vLLM 0.30.0, vLLM-GR dda07ad /
> OneRec-1.7B BF16, ModelRunner V2, graphs on, beam width 128, 3 SID tokens,
> 1,024 input tokens, 100,000-SID constraint table, one run of 30
> measured batches per point. Not a performance guarantee.

| Batch size | Default p50 (ms) | Batch-invariant p50 (ms) | Change |
| --- | ---: | ---: | ---: |
| 1 | 36.2 | 50.2 | +39% |
| 4 | 99.3 | 124.6 | +25% |
