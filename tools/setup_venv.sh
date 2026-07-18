#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$here/.venv"

python3 --version

if [[ ! -d "$venv" ]]; then
	echo "creating venv at $venv"
	python3 -m venv "$venv"
fi

# shellcheck disable=SC1091
source "$venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$here/requirements.txt"

echo "done. dumper interpreter: $venv/bin/python"
