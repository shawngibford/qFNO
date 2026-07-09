using Yao, YaoBlocks, Lux, Random, LinearAlgebra
using Zygote, ChainRulesCore

# ---------------------------------------------------------------------------
# Quantum spectral layer, following the quantum-Fourier-network idea of
# Jain et al. (2024, Quantum Sci. Technol.):
#
#   field vector (length 2^n)  --amplitude encode-->  |ψ⟩ on n qubits
#      --QFT-->  frequency amplitudes
#      --parameterized circuit U(θ)-->  learned spectral filter
#      --inv-QFT-->  back to signal space
#      --measure amplitudes-->  output vector
#
# Everything here runs on Yao's classical statevector simulator. The variational
# block U(θ) is a hardware-efficient ansatz (Ry rotations + CNOT ring) whose depth
# is a knob for the barren-plateau study (Larocca et al. 2025).
# ---------------------------------------------------------------------------

"""
    variational_filter(n, depth) -> Yao chunk

Hardware-efficient ansatz on `n` qubits, `depth` layers.
Each layer: Ry on every qubit, then a CNOT entangling ring.
Parameter count = n * depth.
"""
function variational_filter(n::Int, depth::Int)
    circ = chain(n)
    for _ in 1:depth
        push!(circ, chain(n, [put(n, i=>Ry(0.0)) for i in 1:n]...))
        push!(circ, chain(n, [cnot(n, i, i%n+1) for i in 1:n]...))
    end
    return circ
end

"""
    qft_block(n) -> Yao QFT chunk
"""
qft_block(n::Int) = Yao.EasyBuild.qft_circuit(n)

"""
    run_quantum_filter(vec, θ, n, depth) -> Vector{Float32}

Pure-simulation forward pass for ONE input vector of length 2^n.
Amplitude-encode (normalize), QFT, apply U(θ), inverse-QFT, return |amplitudes|
scaled back to the input norm. `θ` has length n*depth.
"""
function run_quantum_filter(vec::AbstractVector{<:Real}, θ::AbstractVector,
                            n::Int, depth::Int)
    @assert length(vec) == 2^n "vector length $(length(vec)) != 2^$n"
    nrm = norm(vec) + 1f-8
    ψ = ArrayReg(ComplexF64.(vec ./ nrm))
    filt = variational_filter(n, depth)
    dispatch!(filt, collect(Float64.(θ)))            # load trainable angles
    circuit = chain(n, qft_block(n), filt, qft_block(n)')   # QFT, U(θ), inv-QFT
    ψ |> circuit
    amp = state(ψ)[:, 1]
    return Float32.(real.(amp) .* nrm)               # real readout, rescaled
end

# ---- Lux layer wrapping the quantum filter as a channel-wise spectral conv ----

struct QuantumSpectralConv1d <: Lux.AbstractLuxLayer
    n::Int          # qubits; spatial length must be 2^n
    depth::Int      # ansatz depth
    channels::Int   # apply an independent filter per channel
end

function Lux.initialparameters(rng::AbstractRNG, l::QuantumSpectralConv1d)
    # θ shape (n*depth, channels)
    θ = 0.1f0 .* randn(rng, Float32, l.n*l.depth, l.channels)
    return (; θ)
end
Lux.initialstates(::AbstractRNG, ::QuantumSpectralConv1d) = NamedTuple()

# x: (nx=2^n, channels, batch)
function (l::QuantumSpectralConv1d)(x, ps, st)
    nx, ch, nb = size(x)
    @assert nx == 2^l.n "spatial length $nx must equal 2^$(l.n)"
    y = similar(x)
    for b in 1:nb, c in 1:ch
        y[:, c, b] = run_quantum_filter(view(x, :, c, b), view(ps.θ, :, c),
                                        l.n, l.depth)
    end
    return y, st
end

# Custom rrule: parameter-shift-free path via finite differences on θ.
# (For simulation we can also let Zygote differentiate Yao directly; this rrule
#  is the robust fallback and doubles as the parameter-shift hook for hardware.)
function ChainRulesCore.rrule(l::QuantumSpectralConv1d, x, ps, st)
    y, _ = l(x, ps, st)
    function pb(ȳ)
        ȳa = ȳ isa Tuple ? ȳ[1] : ȳ
        nx, ch, nb = size(x)
        ε = 1f-3
        gθ = zeros(Float32, size(ps.θ))
        for c in 1:ch, k in 1:size(ps.θ,1)
            θp = copy(ps.θ); θp[k,c] += ε
            θm = copy(ps.θ); θm[k,c] -= ε
            acc = 0f0
            for b in 1:nb
                yp = run_quantum_filter(view(x,:,c,b), view(θp,:,c), l.n, l.depth)
                ym = run_quantum_filter(view(x,:,c,b), view(θm,:,c), l.n, l.depth)
                acc += sum(((yp .- ym) ./ (2ε)) .* view(ȳa,:,c,b))
            end
            gθ[k,c] = acc
        end
        return (NoTangent(), ZeroTangent(), (; θ=gθ), NoTangent())
    end
    return y, pb
end

"""
    HybridQFNO1d(; n=6, depth=4, ch=8, ch_in=2, ch_out=1)

Hybrid model: classical lift/project (cheap, keeps params trainable) with the
quantum spectral layer doing the global mixing. Spatial length must be 2^n.
"""
function HybridQFNO1d(; n::Int=6, depth::Int=4, ch::Int=8, ch_in::Int=2, ch_out::Int=1)
    return Lux.Chain(
        Lux.Conv((1,), ch_in => ch),                 # classical lift
        QuantumSpectralConv1d(n, depth, ch),         # quantum global mixing
        Lux.WrappedFunction(x -> Lux.gelu.(x)),
        Lux.Conv((1,), ch => 64), Lux.WrappedFunction(x->Lux.gelu.(x)),
        Lux.Conv((1,), 64 => ch_out),                # classical project
    )
end

function build_quantum_qfno(rng::AbstractRNG; kwargs...)
    m = HybridQFNO1d(; kwargs...)
    ps, st = Lux.setup(rng, m)
    return m, ps, st
end
