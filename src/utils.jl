using Statistics, LinearAlgebra, Random

"""
    make_grid(n) -> Vector{Float32}

Uniform grid on [0,1] with `n` points, used as the FNO positional-encoding
channel (Li et al. 2021 append the grid coordinate to the input).
"""
make_grid(n::Int) = collect(range(0f0, 1f0; length=n))

"""
    relative_l2(pred, y) -> Float32

Mean relative L2 error over a batch — the metric FNO and PDEBench report.
`pred`, `y` are (spatial..., batch) or (spatial..., channels, batch); the
reduction is over everything except the last (batch) dimension.
"""
function relative_l2(pred::AbstractArray, y::AbstractArray)
    @assert size(pred) == size(y) "shape mismatch $(size(pred)) vs $(size(y))"
    nb = size(y)[end]
    p = reshape(pred, :, nb)
    t = reshape(y, :, nb)
    num = sqrt.(sum(abs2, p .- t; dims=1))
    den = sqrt.(sum(abs2, t; dims=1)) .+ 1f-8
    return Float32(mean(num ./ den))
end

"""
    normalize_data(x) -> (xn, μ, σ)

Channel/global z-score normalization; returns stats for inversion.
"""
function normalize_data(x::AbstractArray)
    μ = Float32(mean(x)); σ = Float32(std(x)) + 1f-8
    return (x .- μ) ./ σ, μ, σ
end
denormalize(xn, μ, σ) = xn .* σ .+ μ

"count trainable parameters in a Lux/NamedTuple parameter tree"
function count_params(ps)
    total = 0
    function rec(p)
        if p isa AbstractArray
            total += length(p)
        elseif p isa NamedTuple || p isa Tuple
            for v in p; rec(v); end
        end
    end
    rec(ps); return total
end
