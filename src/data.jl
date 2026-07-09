using MAT, HDF5, Random

# ---------------------------------------------------------------------------
# Real benchmark datasets used across the FNO / QFNO literature.
#
#  (A) FNO author files (Li et al. 2021), Google-Drive distributed as .mat:
#        burgers_data_R10.mat        -> 1D Burgers, keys "a" (input) "u" (output)
#        piececonst_r241_N1024_smooth1.mat -> 2D Darcy, keys "coeff" "sol"
#      See https://github.com/neuraloperator/neuraloperator (data links in README).
#
#  (B) PDEBench (Takamoto et al. 2022), HDF5 from the DaRUS repository
#        doi:10.18419/darus-2986 . 1D files store a "tensor" dataset of shape
#        (nsamples, ntime, nx) plus coordinate datasets.
# ---------------------------------------------------------------------------

"""
    load_burgers1d(path; ntrain, ntest, sub=1) -> (xtr,ytr,xte,yte)

Load the FNO 1D Burgers file. Input a(x) (initial condition), output u(x) at
final time. Returns arrays shaped (nx, 1, batch) with the grid appended as a
second channel by the model, or you can append here — kept single-channel so
the model owns positional encoding. `sub` spatially subsamples (stride) to fit
a target grid size; for the quantum model choose sub so nx is a power of two.
"""
function load_burgers1d(path::String; ntrain=1000, ntest=200, sub::Int=1)
    d = matread(path)
    a = Float32.(d["a"]); u = Float32.(d["u"])   # (N, nx)
    a = a[:, 1:sub:end]; u = u[:, 1:sub:end]
    N, nx = size(a)
    @assert ntrain+ntest <= N "requested $(ntrain+ntest) > available $N"
    # reshape to (nx, 1, batch)
    reshb(m, idx) = reshape(permutedims(m[idx, :], (2,1)), nx, 1, length(idx))
    tr = 1:ntrain; te = ntrain+1:ntrain+ntest
    return reshb(a,tr), reshb(u,tr), reshb(a,te), reshb(u,te)
end

"""
    load_darcy2d(path; ntrain, ntest, sub=1) -> (xtr,ytr,xte,yte)

FNO 2D Darcy. coeff -> sol. Returns (nx, ny, 1, batch).
"""
function load_darcy2d(path::String; ntrain=1000, ntest=100, sub::Int=1)
    d = matread(path)
    a = Float32.(d["coeff"]); u = Float32.(d["sol"])   # (N, nx, ny)
    a = a[:, 1:sub:end, 1:sub:end]; u = u[:, 1:sub:end, 1:sub:end]
    N, nx, ny = size(a)
    tr = 1:ntrain; te = ntrain+1:ntrain+ntest
    reshb(m, idx) = reshape(permutedims(m[idx, :, :], (2,3,1)), nx, ny, 1, length(idx))
    return reshb(a,tr), reshb(u,tr), reshb(a,te), reshb(u,te)
end

"""
    load_pdebench_1d(path; t_in=1, t_out=end, ntrain, ntest, sub=1)

Generic PDEBench 1D loader (Advection / Burgers / diffusion-reaction). Reads the
"tensor" dataset (nsamples, ntime, nx). Learns the map u(t_in) -> u(t_out).
"""
function load_pdebench_1d(path::String; t_in::Int=1, t_out::Int=0,
                          ntrain=900, ntest=100, sub::Int=1)
    local a, u, nx
    h5open(path, "r") do f
        T = read(f["tensor"])              # (nsamples, ntime, nx)
        nt = size(T, 2)
        to = t_out == 0 ? nt : t_out
        a = Float32.(T[:, t_in, 1:sub:end])
        u = Float32.(T[:, to,  1:sub:end])
        nx = size(a, 2)
    end
    N = size(a, 1)
    tr = 1:ntrain; te = ntrain+1:ntrain+ntest
    reshb(m, idx) = reshape(permutedims(m[idx, :], (2,1)), nx, 1, length(idx))
    return reshb(a,tr), reshb(u,tr), reshb(a,te), reshb(u,te)
end

"stride-subsample the spatial dim of an (nx,...,batch) array to `target` points"
function subsample(x::AbstractArray, target::Int)
    nx = size(x, 1)
    stride = max(1, nx ÷ target)
    idx = 1:stride:nx
    idx = idx[1:min(target, length(idx))]
    return x[idx, ntuple(_->Colon(), ndims(x)-1)...]
end
