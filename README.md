# QFNO — Classical vs. Quantum Fourier Neural Operators for PDE Surrogates

A controlled, reproducible comparison of a classical Fourier Neural Operator
(FNO) against a hybrid **quantum** FNO whose spectral-mixing layer is a
parameterized quantum circuit simulated with Yao.jl. Built to run on real
benchmark PDE datasets from the operator-learning literature.

## What this measures

The classical FNO is a strong, well-benchmarked surrogate. The quantum FNO is a
promising but simulator-bound idea whose advantage is asymptotic and whose
scaling is threatened by (i) data-loading cost and (ii) barren plateaus. So the
experiment reports three things, not one:

1. **Accuracy** — relative-L2 test error, matched parameter budget and retained-mode count.
2. **Trainability** — gradient variance of the quantum layer vs. qubit count (`barren_plateau.jl`).
3. **Cost accounting** — parameter counts and the encoding-cost caveat, stated explicitly.

See `QFNO_literature_review.md` (in the project) for the grounding and citations.

## Layout

```
Project.toml                 deps: NeuralOperators, Lux, Yao, Zygote, FFTW, HDF5, MAT, Makie
src/
  QFNO.jl                    module entry, include order
  utils.jl                   grid, relative-L2, normalization, param count
  data.jl                    loaders: FNO .mat (Burgers/Darcy) + PDEBench HDF5
  models_classical.jl        from-scratch SpectralConv1d + FNO  (+ NeuralOperators.jl hook)
  models_quantum.jl          QuantumSpectralConv1d (QFT + variational filter, Yao) + Hybrid model
  train.jl                   Adam+WD training loop, rel-L2 loss
  evaluate.jl                test metrics + barren_plateau_scan
scripts/
  smoke_test.jl              synthetic data, no download — run this FIRST
  download_data.jl           where to get the real datasets
  run_experiment.jl          head-to-head from a TOML config
  barren_plateau.jl          trainability scan
configs/burgers.toml         example config
```

## Quick start

**One command** — finds/uses Julia, builds a project-local environment, runs the smoke test:

```bash
./run_smoke.sh
```

This creates a self-contained depot at `./.julia_local` (the venv equivalent —
packages install there, not in your global `~/.julia`), resolves + precompiles
the stack, writes a pinned `Manifest.toml`, then runs `smoke_test.jl` on
synthetic data (no downloads). First run is slow (Yao + Lux precompile);
re-runs are fast. If you don't have Julia, the script tells you how to install
it (`curl -fsSL https://install.julialang.org | sh`).

**Manual equivalent**, if you prefer to drive it yourself:

```bash
julia --project=. scripts/bootstrap.jl                  # resolve + install + precompile
julia --project=. scripts/smoke_test.jl                 # verify the stack (no data needed)
```

**Then the real experiment:**

```bash
julia --project=. scripts/download_data.jl              # dataset pointers
# put burgers_data_R10.mat in data/, then:
julia --project=. scripts/run_experiment.jl configs/burgers.toml
julia --project=. scripts/barren_plateau.jl
```

## The quantum spectral layer

`QuantumSpectralConv1d(n, depth, channels)` implements, per channel, per sample:
amplitude-encode the length-`2^n` field into `n` qubits → QFT → hardware-efficient
variational circuit `U(θ)` (the learnable filter, `n·depth` angles) → inverse-QFT →
real amplitude readout, rescaled to the input norm. This follows the
quantum-Fourier-network construction of Jain et al. (2024). Gradients flow either
through Yao's built-in circuit autodiff or the finite-difference `rrule` provided
(which doubles as the parameter-shift hook for real hardware).

## Design options (pick per experiment)

The scaffold is deliberately modular so you can swap any one axis and hold the
rest fixed. The main knobs, and the trade-off each carries:

| Axis | Options | Trade-off |
|------|---------|-----------|
| **Classical baseline** | from-scratch `ClassicalFNO1d` (default, line-for-line comparable) · `NeuralOperators.jl` library FNO (the "official" baseline) | comparability vs. authority of the baseline |
| **Quantum data encoding** | amplitude encoding (default, `2^n` values in `n` qubits, but `O(2^n)` load) · angle encoding (`n` values in `n` qubits, cheap load, less expressive) | expressivity vs. honest loading cost |
| **Variational ansatz** | hardware-efficient Ry+CNOT ring (default) · alternating-layer / brick-wall · problem-inspired | trainability vs. expressivity (barren plateaus) |
| **Differentiation** | finite-difference `rrule` (default, robust) · Yao native autodiff (faster in sim) · parameter-shift (hardware-ready) | speed vs. hardware fidelity |
| **Dataset** | FNO Burgers 1D (default, smallest) · FNO Darcy 2D · PDEBench Advection/Burgers/diff-react | qubit budget vs. difficulty |
| **Model form** | hybrid (classical lift/project + quantum mixing, default) · fully-quantum readout · quantum-augmented (quantum layer added to classical FNO) | feasibility vs. "how quantum" the claim is |

Recommended progression: **smoke test → Burgers 1D at n=6 → barren-plateau scan → Darcy/PDEBench.**
Start where a simulator is comfortable (≤10 qubits, small channel count — circuit
sims cost `O(channels · batch)` per forward pass) and scale one axis at a time.
