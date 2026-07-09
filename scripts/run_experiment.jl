#!/usr/bin/env julia
# Head-to-head: classical FNO vs hybrid quantum FNO on a real benchmark dataset.
#
#   julia --project=. scripts/run_experiment.jl configs/burgers.toml
#
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using TOML, Random, JLD2, Printf, CairoMakie
include(joinpath(@__DIR__, "..", "src", "QFNO.jl")); using .QFNO

cfgfile = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "burgers.toml")
cfg = TOML.parsefile(cfgfile)
rng = Random.MersenneTwister(cfg["train"]["seed"])

# --- data ---
gs = cfg["data"]["grid_size"]
d  = cfg["data"]
xtr, ytr, xte, yte = if d["dataset"] == "burgers"
    a,b,c,e = load_burgers1d(d["path"]; ntrain=d["ntrain"], ntest=d["ntest"])
    subsample(a,gs), subsample(b,gs), subsample(c,gs), subsample(e,gs)
elseif d["dataset"] == "pdebench"
    a,b,c,e = load_pdebench_1d(d["path"]; ntrain=d["ntrain"], ntest=d["ntest"])
    subsample(a,gs), subsample(b,gs), subsample(c,gs), subsample(e,gs)
else
    error("dataset $(d["dataset"]) not wired into this script")
end
@printf("data: xtr %s  ytr %s\n", size(xtr), size(ytr))

tcfg = TrainConfig(; epochs=cfg["train"]["epochs"], batch=cfg["train"]["batch"],
                     lr=Float32(cfg["train"]["lr"]), wd=Float32(cfg["train"]["wd"]),
                     seed=cfg["train"]["seed"], log_every=cfg["train"]["log_every"])

# --- classical FNO ---
println("\n=== Classical FNO ===")
cm, cps, cst = build_classical_fno(rng; ch=cfg["classical"]["ch"],
    modes=cfg["classical"]["modes"], width_layers=cfg["classical"]["width_layers"])
cps, cst, chist = train_model!(cm, cps, cst, xtr, ytr; cfg=tcfg, xte=xte, yte=yte)
cres = evaluate_model(cm, cps, cst, xte, yte)
@printf("classical: rel-L2 = %.4e ± %.2e  | params = %d\n",
        cres.rel_l2, cres.rel_l2_std, count_params(cps))

# --- hybrid quantum FNO ---
println("\n=== Hybrid Quantum FNO ===")
qm, qps, qst = build_quantum_qfno(rng; n=cfg["quantum"]["n"],
    depth=cfg["quantum"]["depth"], ch=cfg["quantum"]["ch"])
qps, qst, qhist = train_model!(qm, qps, qst, xtr, ytr; cfg=tcfg, xte=xte, yte=yte)
qres = evaluate_model(qm, qps, qst, xte, yte)
@printf("quantum:   rel-L2 = %.4e ± %.2e  | params = %d\n",
        qres.rel_l2, qres.rel_l2_std, count_params(qps))

# --- save + plot ---
mkpath("results")
@save "results/run.jld2" cfg chist qhist cres qres
fig = Figure(size=(900,380))
ax1 = Axis(fig[1,1], title="Validation rel-L2", xlabel="epoch", ylabel="rel-L2", yscale=log10)
lines!(ax1, chist.epoch, chist.val, label="classical FNO")
lines!(ax1, qhist.epoch, qhist.val, label="quantum FNO")
axislegend(ax1)
ax2 = Axis(fig[1,2], title="Per-sample test error", xlabel="rel-L2", ylabel="count")
hist!(ax2, cres.per_sample, bins=20, label="classical")
hist!(ax2, qres.per_sample, bins=20, label="quantum")
axislegend(ax2)
save("results/comparison.png", fig)
println("\nsaved results/run.jld2 and results/comparison.png")
