#!/usr/bin/env bash
# Greedy token-trajectory parity: llama.cpp oracle vs our CLI, same GGUF/prompt.
#
#   tools/parity_check.sh <model.gguf> "<prompt>" N
#
set -euo pipefail

if [[ $# -ne 3 ]]; then
	echo "usage: $0 <model.gguf> \"<prompt>\" N" >&2
	exit 2
fi

model="$1"
prompt="$2"
n="$3"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
llbin="$here/llama.cpp/build/bin"
cli="$llbin/llama-cli"
tok="$llbin/llama-tokenize"

if [[ ! -x "$cli" || ! -x "$tok" ]]; then
	echo "error: llama.cpp oracle not built. Run tools/build_llamacpp.sh first." >&2
	exit 1
fi
if [[ ! -f "$model" ]]; then
	echo "error: model not found: $model" >&2
	exit 1
fi

engine="$root/build/release/engine"
if [[ ! -x "$engine" ]]; then
	echo "error: engine not built: $engine (cmake --build build/release)." >&2
	exit 1
fi

echo "== oracle (llama.cpp) =="
# our engine keeps an fp32 kv cache, so pin the oracle to fp32 kv too. its
# fp16 default flips near-tie tokens the quant math cannot reproduce.
oracle_text="$("$cli" -m "$model" -p "$prompt" -n "$n" --temp 0 -s 0 \
	-ctk f32 -ctv f32 -no-cnv --no-display-prompt </dev/null 2>/dev/null)"

echo "== ours (dbinfer) =="
ours_text="$("$engine" -m "$model" -p "$prompt" -n "$n" --temp 0 -s 0 </dev/null 2>/dev/null)"
ours_ids="$("$engine" -m "$model" -p "$prompt" -n "$n" --temp 0 -s 0 --print-ids </dev/null 2>/dev/null |
	grep -oE '[0-9]+' | head -n "$n" | tr '\n' ' ')"
n_ids="$(printf '%s\n' "$ours_ids" | wc -w | tr -d ' ')"

# Informational only: re-tokenized oracle ids (see header for the caveat).
oracle_ids="$("$tok" -m "$model" -p "$oracle_text" --ids --no-bos 2>/dev/null |
	tr -d '[]' | tr ',' '\n' | grep -oE '[0-9]+' | head -n "$n" | tr '\n' ' ')"

echo "ours ids:        $ours_ids"
echo "oracle retok ids:$oracle_ids"
if [[ "$ours_ids" != "$oracle_ids" ]]; then
	echo "note: raw-id lists differ; this is the detok/retok artifact if the"
	echo "      continuation below is byte-identical (see script header)."
fi

if [[ "$ours_text" == "$oracle_text" ]]; then
	echo "PARITY OK ($n_ids tokens, continuation byte-identical)"
	exit 0
fi
echo "PARITY MISMATCH" >&2
echo "--- ours ---" >&2
printf '%s\n' "$ours_text" >&2
echo "--- oracle ---" >&2
printf '%s\n' "$oracle_text" >&2
exit 1
