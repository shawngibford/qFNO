#!/usr/bin/env julia
# Pointers to the real benchmark datasets used in the FNO / QFNO literature.
# These sources are gated behind Google-Drive / DaRUS interstitials, so this
# script prints the canonical locations rather than blindly curl-ing them.
#
#   julia scripts/download_data.jl
#
println("""
Benchmark datasets for the classical-vs-quantum FNO comparison
==============================================================

(A) FNO author datasets — Li et al. 2021 (arXiv:2010.08895)
    Repo:  https://github.com/neuraloperator/neuraloperator  (see 'Datasets' in README)
    Files: burgers_data_R10.mat            1D Burgers      keys: a, u
           piececonst_r241_N1024_smooth1.mat 2D Darcy      keys: coeff, sol
           NavierStokes_V1e-5_N1200_T20.mat  2D NS (time)
    -> place under  data/

(B) PDEBench — Takamoto et al. 2022 (arXiv:2210.07182)
    Data (DaRUS):  https://doi.org/10.18419/darus-2986
    Code/loaders:  https://github.com/pdebench/PDEBench
    Recommended 1D starters (HDF5, dataset key "tensor", shape (N, t, x)):
       1D_Advection_Sols_beta*.hdf5
       1D_Burgers_Sols_Nu*.hdf5
       1D_diff-react_*.hdf5
    -> place under  data/

Once a file is in data/, point configs/*.toml 'path' at it and run:
    julia --project=. scripts/run_experiment.jl configs/burgers.toml

Tip: keep grid_size a power of two (32, 64, 128) so the quantum amplitude
encoding onto n = log2(grid_size) qubits is exact.
""")
