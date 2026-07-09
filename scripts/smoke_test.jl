#!/usr/bin/env julia
# No-download smoke test: synthetic 1D data, tiny models, both paths run end to
# end (forward + backward + a few epochs). Run this first to confirm the stack
# is wired correctly before pulling real datasets.
#
#   julia --project=. scripts/smoke_test.jl
#
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random, Printf
include(joinpath(@__DIR__, "..", "src", "QFNO.jl")); using .QFNO

rng = Random.MersenneTwister(0)
nx, N = 32, 120                      # 2^5 grid
# synthetic operator: smooth random input -> its running integral (a real map)
x = zeros(Float32, nx, 1, N)
y = zeros(Float32, nx, 1, N)
for i in 1:N
    k = rand(rng, 1:4)
    f = sin.(k .* range(0,2π,length=nx)) .+ 0.3f0*randn(rng, Float32, nx)
    x[:,1,i] = f
    y[:,1,i] = cumsum(f) ./ nx        # discrete integral
end
xtr,ytr,xte,yte = x[:,:,1:100],y[:,:,1:100],x[:,:,101:end],y[:,:,101:end]

cfg = TrainConfig(epochs=15, batch=20, lr=2f-3, log_every=5)

println("=== classical ===")
cm,cps,cst = build_classical_fno(rng; ch=16, modes=8, width_layers=2)
cps,cst,_ = train_model!(cm,cps,cst,xtr,ytr; cfg=cfg, xte=xte, yte=yte)
@printf("classical rel-L2 = %.4e  params=%d\n", evaluate_model(cm,cps,cst,xte,yte).rel_l2, count_params(cps))

println("=== quantum (n=5, depth=2, ch=4) ===")
qm,qps,qst = build_quantum_qfno(rng; n=5, depth=2, ch=4)
qps,qst,_ = train_model!(qm,qps,qst,xtr,ytr; cfg=cfg, xte=xte, yte=yte)
@printf("quantum rel-L2 = %.4e  params=%d\n", evaluate_model(qm,qps,qst,xte,yte).rel_l2, count_params(qps))
println("\nsmoke test OK")
