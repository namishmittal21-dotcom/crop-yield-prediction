#!/usr/bin/env bash
# One-click setup and launch for macOS / Linux:  ./open_notebook.sh
set -e
cd "$(dirname "$0")"

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 or newer was not found. Install it from https://www.python.org/downloads/ and run this script again."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Setting up a private Python environment for this app (first run only)..."
  "$PY" -m venv .venv
fi
echo "Installing / checking packages (the first run takes a few minutes)..."
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-notebook.txt

echo
echo "Opening the notebook in Jupyter (results are already shown; Run > Run All Cells re-runs it)."
exec .venv/bin/python -m notebook Crops_Prediction.ipynb
