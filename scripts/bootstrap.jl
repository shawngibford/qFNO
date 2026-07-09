#!/usr/bin/env julia
# Resolve + install the QFNO environment into a PROJECT-LOCAL depot.
#
# This is the Julia analogue of creating a venv: all packages land in
# ./.julia_local (set by run_smoke.sh via JULIA_DEPOT_PATH), so nothing touches
# your global ~/.julia and the whole project is self-contained. It also writes a
# pinned Manifest.toml next to Project.toml for reproducible re-installs.
#
# Run indirectly via run_smoke.sh, or directly:
#   JULIA_DEPOT_PATH="$PWD/.julia_local" julia scripts/bootstrap.jl
#
import Pkg

proj = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(proj)

println("• depot(s): ", DEPOT_PATH)
println("• project : ", proj)

# If a Manifest already exists, just instantiate it (fast, reproducible path).
manifest = joinpath(proj, "Manifest.toml")
if isfile(manifest)
    println("• Manifest.toml found — instantiating pinned versions")
    Pkg.instantiate()
else
    println("• no Manifest — resolving from Project.toml [compat] bounds")
    Pkg.resolve()
    Pkg.instantiate()
end

println("• precompiling (first run is slow: Yao + Lux are large)")
Pkg.precompile()

# Report the resolved environment so the user has a record.
println("\n=== resolved environment ===")
Pkg.status()
println("\nBOOTSTRAP_DONE")
