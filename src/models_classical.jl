using Lux, FFTW, Random, Zygote

# ---------------------------------------------------------------------------
# Classical 1D Fourier Neural Operator baseline.
#
# Two ways to get this:
#   (1) Use NeuralOperators.jl's FourierNeuralOperator directly (recommended for
#       the "official" baseline). See build_classical_fno_lib below.
#   (2) A compact from-scratch SpectralConv1d + FNO, so the classical and quantum
#       spectral layers are line-for-line comparable. This is the default used in
#       the head-to-head experiment. See ClassicalFNO1d below.
# ---------------------------------------------------------------------------

# ---- (2) transparent from-scratch spectral conv, matches QuantumSpectralConv1d API ----

struct SpectralConv1d <: Lux.AbstractLuxLayer
    ch_in::Int
    ch_out::Int
    modes::Int          # number of low-frequency Fourier modes kept
end

function Lux.initialparameters(rng::AbstractRNG, l::SpectralConv1d)
    scale = 1f0 / (l.ch_in * l.ch_out)
    # complex weights stored as (real, imag), shape (ch_in, ch_out, modes)
    wr = scale .* randn(rng, Float32, l.ch_in, l.ch_out, l.modes)
    wi = scale .* randn(rng, Float32, l.ch_in, l.ch_out, l.modes)
    return (; wr, wi)
end
Lux.initialstates(::AbstractRNG, ::SpectralConv1d) = NamedTuple()

# x: (nx, ch_in, batch)
# Zygote cannot differentiate through array mutation, so the mode-mixing is
# written functionally: build each retained mode's (ch_out, batch) slice and
# concatenate, then zero-pad the untouched high frequencies.
function (l::SpectralConv1d)(x, ps, st)
    nx, _, nb = size(x)
    xperm = permutedims(x, (2,1,3))                 # (ch_in, nx, batch)
    x̂ = rfft(xperm, 2)                               # (ch_in, nfreq, batch) complex
    nfreq = size(x̂, 2)
    W = complex.(ps.wr, ps.wi)                      # (ch_in, ch_out, modes)
    m = min(l.modes, nfreq)
    # per retained mode k: (ch_out, ch_in) * (ch_in, batch) -> (ch_out, batch)
    mixed = [transpose(W[:, :, k]) * x̂[:, k, :] for k in 1:m]   # vector of (ch_out, nb)
    low = reduce((a,b)->cat(a,b; dims=2),
                 (reshape(mk, l.ch_out, 1, nb) for mk in mixed)) # (ch_out, m, nb)
    out = if m < nfreq
        pad = zeros(ComplexF32, l.ch_out, nfreq - m, nb)
        cat(low, pad; dims=2)
    else
        low
    end
    y = irfft(out, nx, 2)                            # (ch_out, nx, batch)
    return permutedims(y, (2,1,3)), st
end

"""
    ClassicalFNO1d(; ch=32, modes=16, width_layers=4, ch_in=2, ch_out=1)

Standard FNO: lift -> [SpectralConv + pointwise linear + GELU] xL -> project.
`ch_in=2` expects channel 1 = field, channel 2 = grid coordinate.
"""
function ClassicalFNO1d(; ch::Int=32, modes::Int=16, width_layers::Int=4,
                          ch_in::Int=2, ch_out::Int=1)
    spectral_block(i) = Lux.Parallel(+,
        SpectralConv1d(ch, ch, modes),
        Lux.Conv((1,), ch => ch))     # 1x1 conv = pointwise linear residual
    blocks = [Lux.Chain(spectral_block(i),
                        Lux.WrappedFunction(x -> Lux.gelu.(x))) for i in 1:width_layers]
    return Lux.Chain(
        Lux.Conv((1,), ch_in => ch),                 # lift
        blocks...,
        Lux.Conv((1,), ch => 128), Lux.WrappedFunction(x->Lux.gelu.(x)),
        Lux.Conv((1,), 128 => ch_out),               # project
    )
end

"""
    build_classical_fno(rng; kwargs...) -> (model, ps, st)
"""
function build_classical_fno(rng::AbstractRNG; kwargs...)
    m = ClassicalFNO1d(; kwargs...)
    ps, st = Lux.setup(rng, m)
    return m, ps, st
end

# ---- (1) library baseline via NeuralOperators.jl (optional) ----
# Uncomment when NeuralOperators.jl is installed; API tracks the SciML docs.
#
# using NeuralOperators
# build_classical_fno_lib(rng; ch=32, modes=16, ch_in=2, ch_out=1) = begin
#     m = FourierNeuralOperator(ch_in => ch_out; modes=(modes,), chs=(ch,ch,ch,ch))
#     ps, st = Lux.setup(rng, m); (m, ps, st)
# end
