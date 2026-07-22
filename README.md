# dbinfer

The fan on my laptop knows this project by sound at this point. It spins up every time a forward pass starts, and it has probably processed more attention weights than I ever will.

dbinfer is a small, hand-written C++23 inference engine that loads GGUF models and generates text directly on Apple Silicon CPU. There’s no server, no external dependencies, and no cloud bill at the end of the month. You pass it a model file and a prompt, and it computes tokens one by one on the same CPU cores running your browser.

### Why build this?

Obviously, `llama.cpp` already exists. It’s faster, far more mature, and written by people with way more experience in this domain.

I built dbinfer anyway because I wanted to understand exactly what happens, tensor by tensor, between taking in a prompt and spitting out tokens. I also used this project as an experiment to test my AI-assisted engineering skills—using Claude extensively throughout the build process to reason through math, debug subtle memory alignment issues, and accelerate iteration.

That said, I only trust an implementation when every number matches reality. Every kernel here was validated against reference tensors from PyTorch before touching the forward pass, and greedy decoding is checked token-by-token against `llama.cpp`. If a check fails, the code is broken until proven otherwise.

This is a personal project hacked together in my free time, so treat it like an open workshop. Parts of it are still pretty rough around the edges, but it's been an incredible way to peek under the hood of modern LLM architectures.

## Features

- **GGUF Loading:** Supports Qwen2.5, Llama 3.2, TinyLlama, and Qwen3. Weights are memory-mapped (`mmap`) straight from disk so they aren't copied into secondary buffers.
- **Transformer Forward Pass:** Runs fully on CPU (RMSNorm, RoPE, grouped-query attention, SwiGLU). All tensor shapes and hyperparameters are parsed directly from file metadata rather than hardcoded by model name.
- **Full Sampling Pipeline:** Supports repetition/frequency/presence penalties, temperature, top-k, top-p, and min-p (greedy decoding applies when temperature is set to zero).
- **KV Cache Options:** Offers three cache backends: a standard dense cache, a sliding-window ring buffer with attention sinks for long contexts, and an int8-quantized variant to keep memory usage down.
- **Prefix Caching:** Saves and loads prompt prefix caches to avoid re-evaluating long static system prompts.
- **Constrained Decoding:** Token-by-token guidance using GBNF grammars.
- **Speculative Decoding:** Supports greedy speculative decoding using a smaller draft model to propose tokens for a larger target model.
- **Metal Acceleration:** Allows offloading transformer layers to Apple Silicon GPUs, using the CPU path as the ground-truth reference implementation.

What it **doesn't** do: host an HTTP API, manage model downloads, or replace production inference engines. It is simply a single binary that runs a prompt through a model locally.

## Building

Requires **CMake 3.24+** and a C++23-compliant Clang compiler. The only external frameworks used are macOS built-ins (`Accelerate` and `Metal`).

### Release Build

```bash
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j

```

This produces the main binary at `build/release/engine`. Metal support is enabled automatically on macOS (disable it with `-DDBINFER_METAL=OFF`).

### Debug & Sanitizers

I use ASan before committing memory-touching changes:

```bash
cmake -B build/asan -DCMAKE_BUILD_TYPE=Debug -DDBINFER_SANITIZE=ON
cmake --build build/asan -j

```

For thread safety checks, build with `-DDBINFER_TSAN=ON` in a separate build directory (ASan and TSan cannot be combined in a single binary).

## Usage

Basic execution:

```bash
./build/release/engine -m /path/to/model.gguf -p "the tide was going out and" -n 64

```

CLI options:

```
usage: engine -m <model.gguf> -p <prompt> [options]
  -m <path>                GGUF model file (required)
  -p <text>                prompt (required)
  -n <int>                 tokens to generate (default 128)
  --threads <int>          worker threads, 0 = auto (default auto)
  --gpu-layers <int>       offload the first N blocks to Metal (default 0)
  --temp <float>           temperature, 0 = greedy (default 0)
  --top-k <int>            top-k cutoff, 0 = off (default 0)
  --top-p <float>          top-p nucleus, [0,1] (default 1)
  --min-p <float>          min-p cutoff, [0,1] (default 0)
  --repeat-penalty <f>     repetition penalty, >0 (default 1)
  --freq-penalty <f>       frequency penalty (default 0)
  --presence-penalty <f>   presence penalty (default 0)
  --penalty-last-n <int>   penalty window (default 64)
  -s, --seed <uint>        RNG seed
  --print-ids              print token ids instead of text
  --perplexity <path>      score perplexity over a text file (no -p)
  --ppl-chunks <int>       limit perplexity to N windows (default all)
  --ppl-stream             stream one continuous context, bucket by position
  --kv-window <int>        ring-buffer window, 0 = dense cache (default 0)
  --kv-sink <int>          pinned attention sink positions (default 4)
  --kv-int8                store keys per-channel and values per-block as int8
  --kv-cache-save <path>   dump the dense fp32 prefix cache after prefill
  --kv-cache-load <path>   load a prefix cache and continue from it
  --prefill-chunk <int>    prefill tokens per chunk, 1 = per token (default 1)
  --grammar <path>         constrain decoding to a GBNF grammar file
  --grammar-stop           stop when the grammar first reaches a complete state
  --draft-model <path>     draft GGUF for greedy speculative decoding
  --draft-k <int>          draft tokens proposed per verify round (default 4)

```

You can also set the `DBINFER_GPU_LAYERS` environment variable instead of passing `--gpu-layers` every time. The default temperature is `0` (greedy decoding).

## Testing & Parity Validation

Run the full CTest suite:

```bash
ctest --test-dir build/release --output-on-failure

```

Numerical ops are matched against reference tensors generated via HuggingFace Transformers (`tools/dump_reference.py`):

```bash
tools/setup_venv.sh
tools/.venv/bin/python tools/dump_reference.py --model qwen2.5-0.5b

```

Tolerances are strictly enforced across tests:

- **`1e-5`** max error for single FP32 math operations.
- **`1e-3`** max error across the full FP32 forward pass.
- **`1e-6`** max deviation between SIMD/NEON kernels and scalar code.

### Parity Check

To verify output against `llama.cpp` token-for-token at greedy settings:

```bash
./tools/parity_check.sh <model.gguf> "<prompt>" 64

```

This runs both binaries against the same model/prompt and confirms the generated token IDs match exactly. The exact commit of `llama.cpp` used for testing is pinned in `tools/LLAMACPP_VERSION`.

## Architecture Details

High-level pipeline:

```
prompt -> tokenizer -> embed
       -> [ RMSNorm -> QKV -> RoPE -> attention -> RMSNorm -> SwiGLU ] x n_layers
       -> RMSNorm -> logits -> sampler -> token

```

Key technical choices in the codebase:

- **Tensor Layouts:** GGUF stores dimensions in reverse order relative to standard matrix notation (`[out, in]` appears as `(in, out)` in storage). The loader normalizes these to row-major `[rows=out, cols=in]` during load time so downstream kernels don't need dimension swaps.
- **Zero-Copy Weights:** Weights are referenced directly from the memory-mapped file; dynamic memory allocations are reserved strictly for intermediate activation tensors.
- **Architecture Agnostic:** Modern model quirks are inferred dynamically. Tied embeddings are handled by detecting whether `output.weight` exists (reusing input embeddings if missing), and layer components like QKV bias, norm epsilon, or RoPE theta are read directly from file metadata.
- **Metal GPU Backend:** The Metal pipeline relies on the CPU kernels as the ground truth reference. Every Metal kernel is validated against equivalent CPU tensor outputs before running in the main inference loop.

## Project Structure

```
src/gguf/        GGUF file parsing, mmap handling, and metadata extraction
src/tokenizer/   BPE and SentencePiece tokenizers
src/tensor/      Tensor operations, matrix multiplication, quantization, NEON kernels
src/model/      Forward pass implementation, KV cache variants, speculative decoding
src/sample/     Sampling pipeline (logits filters, top-k/p, min-p, penalties)
src/grammar/    GBNF grammar parser and token masking
src/dbmf/       Custom model format tools and GGUF converter
src/backend/    Metal GPU backend shaders and dispatch code
tests/          CTest unit tests for individual subsystems
tools/          Reference tensor generation, benchmarking, and parity validation

```

## Known Limitations

- **Quantization:** Quantization support is currently limited to `Q8_0` and `Q4_0`.
- **Quantized Parity:** Quantized outputs are not 100% token-identical to `llama.cpp` due to minor differences in float reduction order and KV cache precision. Quantized model accuracy is instead evaluated using perplexity scoring.
- **Benchmarks:** Accurate performance benchmarks are still a work in progress.

If you are looking for code to study how a transformer forward pass works down to the raw matrix operations without framework abstraction, take a look at `src/model/model.cpp`.
