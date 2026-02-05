#!/usr/bin/env bash
# Init the pinned llama.cpp submodule and build the CPU-only oracle binaries.
#
# CPU-only (-DGGML_METAL=OFF) so the parity/perplexity oracle is deterministic
# against our CPU engine (docs/VERIFICATION.md:43-46). The pinned commit and
# flags live in tools/LLAMACPP_VERSION; this script is the single source that
# consumes them.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
sub="$here/llama.cpp"
verfile="$here/LLAMACPP_VERSION"

pin="$(grep -E '^commit=' "$verfile" | head -1 | cut -d= -f2)"
if [[ -z "$pin" ]]; then
	echo "error: no commit= pin in $verfile" >&2
	exit 1
fi
echo "pinned llama.cpp commit: $pin"

# Fetch the submodule contents if not present.
if [[ ! -e "$sub/CMakeLists.txt" ]]; then
	echo "initializing submodule tools/llama.cpp"
	git -C "$root" submodule update --init --recursive tools/llama.cpp
fi

# Check out the pinned commit exactly.
git -C "$sub" fetch --depth 1 origin "$pin" 2>/dev/null || git -C "$sub" fetch origin
git -C "$sub" checkout --detach "$pin"
echo "checked out $(git -C "$sub" rev-parse HEAD)"

# Configure CPU-only and build just the oracle binaries we use.
cmake -S "$sub" -B "$sub/build" \
	-DCMAKE_BUILD_TYPE=Release \
	-DGGML_METAL=OFF \
	-DLLAMA_CURL=OFF \
	-DGGML_NATIVE=ON
cmake --build "$sub/build" -j \
	--target llama-cli llama-tokenize llama-perplexity llama-bench

echo "built oracle binaries into $sub/build/bin"
ls -1 "$sub/build/bin" | grep -E '^llama-(cli|tokenize|perplexity|bench)$' || true
