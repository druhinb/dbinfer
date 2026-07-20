# dbinfer

CPU-first LLM inference engine in C++23 for Apple Silicon. Loads GGUF models,
runs a full transformer forward pass, and generates tokens from a CLI. Greedy
decoding is token-identical to llama.cpp on the same fp32 model (see
`docs/VERIFICATION.md`). No HTTP server.

Ships with quantization (Q8_0/Q4_0/Q5_0/Q4_K/Q6_K), NEON SIMD, multithreading,
a KV cache with attention sinks/int8/prefix-caching/chunked prefill, GBNF
grammar-constrained decoding, speculative decoding, a custom model format
(DBMF) with optional lossless compression, and a Metal GPU backend.

## Build

```bash
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

Debug config with ASan/UBSan: `cmake -B build/asan -DCMAKE_BUILD_TYPE=Debug`.
Metal support builds by default on Apple hardware (`DBINFER_METAL=ON`); pass
`-DDBINFER_METAL=OFF` to disable it.

## Get a model

```bash
tools/build_llamacpp.sh                        # pinned llama.cpp oracle + llama-quantize
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf \
  --local-dir tools/llama.cpp/models
mv tools/llama.cpp/models/qwen2.5-0.5b-instruct-fp16.gguf tools/llama.cpp/models/qwen2.5-0.5b.gguf

# quantize (smaller, faster, small accuracy cost — see docs/VERIFICATION.md for tolerances)
mkdir -p build/quant
./tools/llama.cpp/build/bin/llama-quantize \
  tools/llama.cpp/models/qwen2.5-0.5b.gguf build/quant/qwen2.5-0.5b-q8_0.gguf Q8_0
```

Other quant types: `Q4_0`, `Q4_K_M`, `Q6_K` work the same way. `Q4_K_M` pulls
in `Q5_0`/`Q6_K` too for tensors whose rows aren't 256-divisible — this engine
supports all of them.

## Run

```bash
./build/release/engine -m build/quant/qwen2.5-0.5b-q8_0.gguf -p "your prompt" -n 128
```

## Run at max speed

The fastest configuration measured on this hardware (Apple M3 Max, 12 P-cores)
is Q8_0 quantization with full Metal GPU offload:

```bash
DBINFER_BACKEND=metal ./build/release/engine \
  -m build/quant/qwen2.5-0.5b-q8_0.gguf \
  --gpu-layers 24 --threads 12 \
  -p "your prompt" -n 256
```

`--gpu-layers 24` offloads all transformer blocks (this model has 24) to
Metal; use a smaller number for partial offload. `DBINFER_BACKEND=metal` is
required — the GPU path is opt-in, CPU is the default everywhere.

Measured decode throughput, qwen2.5-0.5b, median of 3, `tools/bench` (see
`BENCH.log` for the full history):

| config | Q8_0 tok/s | F16 tok/s |
|---|---:|---:|
| CPU, threaded (i8mm) | 90 | 8.5 |
| GPU, `--gpu-layers 24` | **147** | 136 |
| llama.cpp Metal (`-ngl 99`, reference) | 246 | 188 |

GPU decode beats this engine's own CPU path on both dtypes and reaches ~60%
(Q8_0) / ~72% (F16) of llama.cpp's Metal backend. The remaining gap is a
launch-latency-bound dispatch chain (~411 GPU dispatches/token at m=1); see
`BENCH.log`'s "Phase 11 M4" section for the full profiling breakdown and
what would close it further.

For **long prompts**, F16 GPU prefill is faster than Q8_0 GPU decode is
optimized for — Q8_0's m>1 (prefill) path isn't yet routed to Metal:

```bash
DBINFER_BACKEND=metal ./build/release/engine \
  -m tools/llama.cpp/models/qwen2.5-0.5b.gguf \
  --gpu-layers 24 --prefill-chunk 32 \
  -p "$(cat long_prompt.txt)" -n 128
```

Verify a config still produces correct output before trusting it for real
work — greedy decode with the same seed should be at least perplexity-close
to the CPU path (`--perplexity <wikitext file>`), and F16 GPU decode is
token-identical to CPU:

```bash
diff <(./build/release/engine -m <model> -p "prompt" -n 64 --temp 0 -s 0 --print-ids) \
     <(DBINFER_BACKEND=metal ./build/release/engine -m <model> -p "prompt" -n 64 --temp 0 -s 0 --gpu-layers 24 --print-ids)
```

## Other features

Full flag list: `./build/release/engine` with no args prints usage.

- **Threading**: `--threads N` (default: auto-detect P-cores). CPU decode is
  bitwise-deterministic across thread counts.
- **KV cache**: `--kv-window N --kv-sink K` for a StreamingLLM ring buffer
  (bounded memory, flat long-generation perplexity); `--kv-int8` for
  per-channel-quantized keys + fp32 sink retention (~0.07% perplexity cost);
  `--kv-cache-save <path>` / `--kv-cache-load <path>` to checkpoint and resume
  a prefix without recomputation; `--prefill-chunk N` for batched prefill.
- **Grammar-constrained decoding**: `--grammar grammars/json.gbnf` masks
  logits to a GBNF grammar every step. `--grammar-stop` stops generation once
  the grammar reaches a complete state.
- **Speculative decoding**: `--draft-model <smaller.gguf> --draft-k 4` — a
  smaller/quantized draft model proposes tokens, the loaded model verifies;
  output is token-identical to running the target alone.
- **DBMF** (this project's own model container, research-informed — see
  `docs/ROADMAP.md` Phase 10): page-aligned for Metal zero-copy, per-tensor
  integrity checksums, optional lossless F16 compression (~15% smaller,
  bit-exact on load).
  ```bash
  ./build/release/tools/gguf_to_dbmf [--compress] in.gguf out.dbmf
  ./build/release/engine -m out.dbmf -p "prompt"   # loader sniffs the format
  ```
- **Perplexity**: `--perplexity <text file> --ppl-chunks N` scores a corpus
  against the loaded model (WikiText-2 is the project's standing benchmark;
  fetch with `tools/llama.cpp/scripts/get-wikitext-2.sh`).

## Docs

- `docs/ARCHITECTURE.md` — forward pass spec, KV cache layout, tensor
  conventions, backend interface.
- `docs/VERIFICATION.md` — test strategy, tolerances, parity/perplexity
  methodology.
- `docs/ROADMAP.md` — phase-by-phase status and acceptance criteria.
- `BENCH.log` — every throughput/parity/perplexity measurement this project
  has recorded, with commit hashes.
