#!/usr/bin/env bash
# Greedy token-ID parity: llama.cpp oracle vs our CLI, same GGUF/prompt.
#
#   tools/parity_check.sh <model.gguf> "<prompt>" N
#
# Runs the pinned llama.cpp oracle at temp 0 (docs/VERIFICATION.md:34-46),
# extracts the N continuation token IDs, and diffs them against our engine's
# IDs. Raw prompt, NO chat template (parity caveat, VERIFICATION.md:43-46).
#
# Phase 0: our engine does not exist yet. The oracle half runs; our half is a
# clean stub that exits with a "engine not built (Phase 1)" notice. This script
# becomes a real two-sided diff once the Phase 1 CLI lands.
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

echo "== oracle (llama.cpp) =="
# --no-display-prompt so stdout is only the continuation text. Raw prompt.
oracle_text="$("$cli" -m "$model" -p "$prompt" -n "$n" --temp 0 -s 0 \
	--no-display-prompt 2>/dev/null)"

# Turn the continuation text into token IDs, keep the first N.
oracle_ids="$("$tok" -m "$model" -p "$oracle_text" 2>/dev/null |
	grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ' | head -n "$n" | tr '\n' ' ')"
echo "oracle ids: $oracle_ids"

# --- Our engine (Phase 1+) --------------------------------------------------
engine="$root/build/release/engine"
if [[ ! -x "$engine" ]]; then
	echo
	echo "engine not built yet (Phase 1): $engine absent."
	echo "Oracle side ran; our-CLI side is stubbed until the engine exists."
	echo "Parity diff will run once Phase 1 lands the CLI."
	exit 0
fi

echo "== ours (dbinfer) =="
ours_ids="$("$engine" -m "$model" -p "$prompt" -n "$n" --temp 0 --print-ids 2>/dev/null |
	grep -oE '[0-9]+' | head -n "$n" | tr '\n' ' ')"
echo "ours ids:   $ours_ids"

if [[ "$oracle_ids" == "$ours_ids" ]]; then
	echo "PARITY OK ($n tokens identical)"
	exit 0
fi
echo "PARITY MISMATCH" >&2
exit 1
