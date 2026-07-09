#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QFNO one-shot setup + smoke test.
#
#   ./run_smoke.sh
#
# What it does, in order:
#   1. Find a Julia >= 1.10 (or tell you how to install one).
#   2. Create a PROJECT-LOCAL depot at ./.julia_local  (the venv equivalent:
#      packages install here, not in your global ~/.julia).
#   3. Resolve + install + precompile the environment (scripts/bootstrap.jl),
#      writing a pinned Manifest.toml.
#   4. Run the smoke test (synthetic data, no downloads).
#
# Re-running is cheap: steps 2-3 are skipped once the depot is populated.
# Env overrides:  JULIA_BIN=/path/to/julia  ./run_smoke.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- project-local depot (self-contained, like a venv) ----
export JULIA_DEPOT_PATH="$HERE/.julia_local"
export JULIA_PROJECT="$HERE"
mkdir -p "$JULIA_DEPOT_PATH"

# ---- 1. locate Julia ----
find_julia() {
  if [ -n "${JULIA_BIN:-}" ] && [ -x "${JULIA_BIN}" ]; then echo "$JULIA_BIN"; return; fi
  if command -v julia >/dev/null 2>&1; then command -v julia; return; fi
  # juliaup default install location
  for c in "$HOME/.juliaup/bin/julia" "$HOME/.local/bin/julia"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo ""
}

JULIA="$(find_julia)"
if [ -z "$JULIA" ]; then
  cat <<'MSG'
No Julia found on PATH.

Install it (recommended: juliaup), then re-run this script:

  curl -fsSL https://install.julialang.org | sh
  # restart your shell, then:
  juliaup add 1.10 && juliaup default 1.10

Or set JULIA_BIN to an existing binary:
  JULIA_BIN=/opt/julia-1.10/bin/julia ./run_smoke.sh
MSG
  exit 1
fi

echo "==> using Julia: $JULIA"
"$JULIA" --version

# ---- 2+3. bootstrap the environment into the local depot ----
echo "==> bootstrapping environment into $JULIA_DEPOT_PATH"
"$JULIA" --startup-file=no --project="$HERE" scripts/bootstrap.jl

# ---- 4. smoke test ----
echo "==> running smoke test"
"$JULIA" --startup-file=no --project="$HERE" scripts/smoke_test.jl

echo
echo "==> smoke test finished. Next:"
echo "    $JULIA --project=. scripts/download_data.jl"
echo "    $JULIA --project=. scripts/run_experiment.jl configs/burgers.toml"
echo "    $JULIA --project=. scripts/barren_plateau.jl"
