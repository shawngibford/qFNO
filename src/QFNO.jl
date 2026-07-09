module QFNO

# Shared utilities, data loaders, models, training, evaluation.
# Load order matters: utils -> data -> models -> train -> eval.

include("utils.jl")
include("data.jl")
include("models_classical.jl")
include("models_quantum.jl")
include("train.jl")
include("evaluate.jl")

export make_grid, relative_l2, normalize_data, denormalize
export load_burgers1d, load_darcy2d, load_pdebench_1d, subsample
export ClassicalFNO1d, build_classical_fno
export QuantumSpectralConv1d, HybridQFNO1d, build_quantum_qfno
export train_model!, TrainConfig
export evaluate_model, barren_plateau_scan, count_params

end # module
