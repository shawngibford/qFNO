using Lux, Zygote, Optimisers, Random, Printf, Statistics

Base.@kwdef struct TrainConfig
    epochs::Int      = 100
    batch::Int       = 20
    lr::Float32      = 1f-3
    wd::Float32      = 1f-4
    seed::Int        = 0
    log_every::Int   = 5
end

"append the grid coordinate channel: (nx,1,batch) -> (nx,2,batch)"
function add_grid(x)
    nx, _, nb = size(x)
    g = reshape(make_grid(nx), nx, 1, 1)
    return cat(x, repeat(g, 1, 1, nb); dims=2)
end

function iterate_minibatches(x, y, bs, rng)
    n = size(x)[end]; idx = shuffle(rng, 1:n)
    return (( selectdim(x, ndims(x), idx[i:min(i+bs-1,n)]),
              selectdim(y, ndims(y), idx[i:min(i+bs-1,n)]) )
            for i in 1:bs:n)
end

"""
    train_model!(model, ps, st, xtr, ytr; cfg, xte=nothing, yte=nothing)
        -> (ps, st, history)

Trains with relative-L2 loss + Adam/weight decay. `xtr` is (nx,1,batch);
the grid channel is appended internally. Returns training/val curves.
"""
function train_model!(model, ps, st, xtr, ytr;
                      cfg::TrainConfig=TrainConfig(), xte=nothing, yte=nothing)
    rng = Random.MersenneTwister(cfg.seed)
    xtr = add_grid(xtr)
    xte === nothing || (xte = add_grid(xte))
    opt = Optimisers.OptimiserChain(Optimisers.WeightDecay(cfg.wd), Optimisers.Adam(cfg.lr))
    stt = Optimisers.setup(opt, ps)
    hist = (train=Float32[], val=Float32[], epoch=Int[])
    lossf(p, xb, yb) = begin
        ŷ, _ = Lux.apply(model, xb, p, st); relative_l2(ŷ, yb)
    end
    for ep in 1:cfg.epochs
        ep_loss = 0f0; nb = 0
        for (xb, yb) in iterate_minibatches(xtr, ytr, cfg.batch, rng)
            l, back = Zygote.pullback(p -> lossf(p, xb, yb), ps)
            gs = back(1f0)[1]
            stt, ps = Optimisers.update(stt, ps, gs)
            ep_loss += l; nb += 1
        end
        if ep % cfg.log_every == 0 || ep == 1
            tr = ep_loss/nb
            push!(hist.train, tr); push!(hist.epoch, ep)
            if xte !== nothing
                v, _ = Lux.apply(model, xte, ps, st)
                vl = relative_l2(v, yte); push!(hist.val, vl)
                @printf("epoch %4d  train %.4e  val %.4e\n", ep, tr, vl)
            else
                @printf("epoch %4d  train %.4e\n", ep, tr)
            end
        end
    end
    return ps, st, hist
end
