#!/usr/bin/env julia
# Trainability diagnostic: gradient variance of the quantum spectral layer vs.
# qubit count (Larocca et al. 2025). Exponential decay = barren plateau.
#
#   julia --project=. scripts/barren_plateau.jl
#
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random, Printf, CairoMakie
include(joinpath(@__DIR__, "..", "src", "QFNO.jl")); using .QFNO

build_fn(n) = build_quantum_qfno(Random.MersenneTwister(rand(1:10^6)); n=n, depth=6, ch=1)
xsample(n) = begin
    v = randn(Float32, 2^n); reshape(v ./ maximum(abs.(v)), 2^n, 1, 1)
end

res = barren_plateau_scan(build_fn, xsample; qubit_range=4:2:10, depth=6, nsamples=40)
ns  = sort(collect(keys(res)))
vs  = [res[n] for n in ns]
for (n,v) in zip(ns,vs); @printf("n=%2d   Var[∂θ] = %.3e\n", n, v); end

mkpath("results")
fig = Figure(size=(560,420))
ax = Axis(fig[1,1], title="Barren-plateau scan", xlabel="qubits n",
          ylabel="Var[∂loss/∂θ]", yscale=log10)
scatterlines!(ax, ns, vs)
save("results/barren_plateau.png", fig)
println("saved results/barren_plateau.png")
