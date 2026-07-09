using Lux, Statistics, Random, Zygote

"""
    evaluate_model(model, ps, st, xte, yte) -> NamedTuple

Test relative-L2 plus per-sample distribution. `xte` is (nx,1,batch); grid is
appended here to match training.
"""
function evaluate_model(model, ps, st, xte, yte)
    nx, _, nb = size(xte)
    g = reshape(make_grid(nx), nx, 1, 1)
    xg = cat(xte, repeat(g,1,1,nb); dims=2)
    ŷ, _ = Lux.apply(model, xg, ps, st)
    per = [relative_l2(reshape(selectdim(ŷ,ndims(ŷ),i),size(ŷ)[1:end-1]...,1),
                       reshape(selectdim(yte,ndims(yte),i),size(yte)[1:end-1]...,1))
           for i in 1:nb]
    return (rel_l2=Float32(mean(per)), rel_l2_std=Float32(std(per)),
            per_sample=per, pred=ŷ)
end

"""
    barren_plateau_scan(build_fn, xsample; qubit_range, depth, nsamples=50)
        -> Dict(n => variance_of_gradient)

Diagnostic from Larocca et al. (2025): sample random parameter sets, compute the
gradient of a single-output loss w.r.t. the quantum angles, and record the
variance of one representative component vs. qubit count. Exponential decay in n
is the barren-plateau signature. `build_fn(n)` must return (model, ps, st) whose
params contain a quantum `θ`; `xsample(n)` returns a matching (2^n,1,1) input.
"""
function barren_plateau_scan(build_fn, xsample; qubit_range=4:2:10,
                             depth::Int=6, nsamples::Int=50, seed::Int=0)
    rng = Random.MersenneTwister(seed)
    out = Dict{Int,Float64}()
    for n in qubit_range
        grads = Float64[]
        for _ in 1:nsamples
            model, ps, st = build_fn(n)
            x = xsample(n)
            g = Zygote.gradient(p -> begin
                    ŷ,_ = Lux.apply(model, x, p, st); sum(ŷ)
                end, ps)[1]
            # pull the quantum θ gradient wherever it lives in the tree
            θg = _find_theta_grad(g)
            θg === nothing || push!(grads, Float64(θg[1]))
        end
        out[n] = isempty(grads) ? NaN : var(grads)
        @info "barren-plateau scan" n var=out[n] samples=length(grads)
    end
    return out
end

function _find_theta_grad(g)
    if g isa NamedTuple
        haskey(g, :θ) && g.θ !== nothing && return g.θ
        for v in values(g)
            r = _find_theta_grad(v); r === nothing || return r
        end
    elseif g isa Tuple
        for v in g; r = _find_theta_grad(v); r === nothing || return r; end
    end
    return nothing
end
