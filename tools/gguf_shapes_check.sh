#!/usr/bin/env bash
set -euo pipefail

# Compares our gguf_dump_tool's tensor name, type, shape
# we report row-major, so the reference dims are reversed before diffing.

if [ $# -ne 1 ]; then
	echo "usage: $0 <model.gguf>" >&2
	exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$model" ]; then
	echo "skip: model not found: $model"
	exit 0
fi

tool=""
for cand in \
	"$repo_root/build/release/tools/gguf_dump_tool" \
	"$repo_root/build/asan/tools/gguf_dump_tool"; do
	if [ -x "$cand" ]; then
		tool="$cand"
		break
	fi
done
if [ -z "$tool" ]; then
	echo "error: gguf_dump_tool not built (build the release preset first)" >&2
	exit 3
fi

py="$repo_root/tools/.venv/bin/python"
dump_py="$repo_root/tools/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py"

ours="$(mktemp)"
theirs="$(mktemp)"
trap 'rm -f "$ours" "$theirs"' EXIT

"$tool" "$model" | awk '/shape=\[/ {
  name=$1; type=$2;
  match($0, /\[[^]]*\]/); s=substr($0, RSTART+1, RLENGTH-2);
  gsub(/ /, "", s);
  print name" "type" "s;
}' | sort >"$ours"

PYTHONPATH="$repo_root/tools/llama.cpp/gguf-py" "$py" "$dump_py" "$model" |
	awk -F'|' 'NF==4 {
      dims=$2; type=$3; name=$4;
      gsub(/^ *| *$/, "", name);
      gsub(/ /, "", type);
      gsub(/ /, "", dims);
      n=split(dims, a, ",");
      while (n>1 && a[n]=="1") n--;
      out="";
      for (i=n; i>=1; i--) out=out a[i] (i>1 ? "," : "");
      print name" "type" "out;
    }' | sort >"$theirs"

if diff -u "$theirs" "$ours"; then
	echo "PASS: $(wc -l <"$ours" | tr -d ' ') tensors match gguf-dump"
	exit 0
else
	echo "FAIL: tensor shape/type mismatch vs gguf-dump" >&2
	exit 1
fi
