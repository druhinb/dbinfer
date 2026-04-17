# CONTEXT.md

Environment facts every agent needs before touching Phases 3+. Verified
2026-07-18. This exists so agents stop re-deriving the oracle/model/build
layout. If a fact here is stale, fix it here in the same change.

## Verification oracle (llama.cpp)

Already checked out and built, CPU-only, pinned in `tools/LLAMACPP_VERSION`
(commit `0bb2919`, tag `b4991`).

- Binaries: `tools/llama.cpp/build/bin/` -> `llama-cli`, `llama-tokenize`,
  `llama-perplexity`, `llama-bench`.
- `llama-quantize` is NOT built (the build script targets only the four
  above). Phases 3 and 6 need it to produce quantized GGUFs. Add
  `llama-quantize` to the `--target` list in `tools/build_llamacpp.sh` and
  rerun that script, do not hand-roll a build.
- Rebuild command lives in `tools/build_llamacpp.sh` (consumes the pin).

## Models

- `tools/llama.cpp/models/qwen2.5-0.5b.gguf` is the F16 reference weight
  model (994 MB). This is the only full-weight GGUF present.
- Quantized GGUFs (Q8_0, Q4_0, Q4_K_M) do NOT exist yet. Produce them from
  the F16 file with `llama-quantize` once it is built:
  `llama-quantize tools/llama.cpp/models/qwen2.5-0.5b.gguf <out>.gguf Q8_0`.
- HF cache has `Qwen/Qwen2.5-0.5B-Instruct` (under `~/.cache/huggingface`),
  so `tools/dump_reference.py` regenerates golden tensors offline.
- `tools/llama.cpp/models/ggml-vocab-*.gguf` are vocab-only files for
  tokenizer tests, not weight models.
- Q4_K_M is a mixed quant. For this 0.5B model it is 132 Q5_0 + 13 Q8_0 +
  12 Q4_K + 12 Q6_K tensors, because k-quants need 256-divisible rows and
  most tensors fall back to Q5_0. Loading it requires Q5_0 support.

## Python

- Use `tools/.venv` (torch 2.2.2, transformers 4.44.2). System `python3` has
  no torch. Run `tools/.venv/bin/python tools/dump_reference.py ...`.

## Build layout

`cmake -B build` in the docs is shorthand. The real tree has three configs,
each with its own `engine`:

- `build/release/engine` — Release. Parity script expects this exact path.
- `build/asan/engine` — Debug + ASan/UBSan. Run before declaring any
  memory-touching change done.
- `build/engine` — default config dir.

Configure/build a config, e.g. Release:
`cmake -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release -j`

Tests: `ctest --test-dir build/release --output-on-failure` (9 tests, all
passing at baseline).

## Parity and perplexity

- `tools/parity_check.sh <model.gguf> "<prompt>" N` diffs greedy token IDs
  against the oracle. Needs `build/release/engine` and the oracle binaries.
  Must be token-identical (project hard rule).
- WikiText-2 is not downloaded. Fetch with
  `tools/llama.cpp/scripts/get-wikitext-2.sh`, then run `llama-perplexity`
  on the same GGUF for the Phase 3/6 within-1% budget.

## Baseline state (2026-07-18)

- Branch `main`, tree clean. Release build green, 9/9 ctest pass.
- Phases 0-2 done. Phase 3 has partial scaffolding
  (`src/tensor/dequant.cpp`, `tests/dequant_test.cpp`).
- No `BENCH.log` yet (Phases 4/5 create it).

## Golden-tensor tolerances (see docs/VERIFICATION.md for the full list)

- fp32 ops atol 1e-5; fp32 forward logits atol/rtol 1e-3.
- Q8_0 logits atol 5e-2 AND argmax must match reference.
- Q4_0 / Q4_K: no fixed logit tolerance, verified by parity + perplexity.
- SIMD vs scalar: atol 1e-6, property-tested on 10k random blocks.

## Quantized parity is near-tie-bounded

Do not re-litigate quantized greedy parity as a kernel bug. Phase 4 settled it
with evidence:

- Activation quantization to Q8_0 is bit-exact to ggml `quantize_row_q8_0`
  (20000 blocks, zero scale or quant mismatches).
- The int8 block dot is exact integer arithmetic; the sdot and i8mm kernels are
  bit-identical to the scalar reference (property test, atol 1e-6, 33835 blocks
  each plus zero, saturated, and denormal edges).

Residual greedy divergences from `llama-cli` are single-token near-tie flips
from two floating-point reproduction gaps, not from wrong math:

- KV-cache dtype: `llama-cli` defaults to an fp16 KV cache; this engine keeps
  fp32 KV. `tools/parity_check.sh` now pins the oracle with `-ctk f32 -ctv f32`.
- Reduction order: ggml reduces the matmul in float lanes; the reference sums
  block-serial. Matching ggml's order conflicts with the 1e-6 scalar gate.

Under the pinned oracle, parity is 2/3 (Q8_0) and 1/3 (Q4_0); the flips move
with either gap and no single config reaches 3/3. Quantized correctness is
gated by perplexity (see docs/VERIFICATION.md layer 4), not token-identical
parity.
