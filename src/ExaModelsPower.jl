module ExaModelsPower

import ExaModels: ExaModels, ExaCore, @add_var, @add_par, @add_con, @add_obj, @add_con!, ExaModel, convert_array, solution
import ExaModels: TwoStageExaCore, EachScenario
using DelimitedFiles
using ExaPowerIO
using JSON

# Parsers and data stay separate from the models that consume them.
include("parser.jl")          # matpower -> the tables every model is built from
include("constraint.jl")      # the algebraic expressions, shared by all three
include("goc3_parser.jl")     # GOC3-specific parsing
include("sc_parser.jl")       #   "

include("opf.jl")             # static: polar, rect, DC
include("mpopf.jl")           # multi-period: polar, rect, DC
include("goc3.jl")            # security-constrained, GOC3 formulation
include("scopf.jl")           # security-constrained, N-1: monolithic + two-stage Schur

const NAMES = filter(names(@__MODULE__; all = true)) do x
    str = string(x)
    endswith(str, "model") && !startswith(str, "#")
end

for name in filter(names(@__MODULE__; all = true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end

# The loop above exports every `*_model` constructor. A model written as a
# recipe has two more halves that a caller needs by name — the recipe itself and
# the arguments that close it — plus the eagerly-built core that `ExaModelsC`
# compiles. Neither ends in "model", so they are named here.
export OPFForm, Polar, Rect, DC
export opf_recipe, opf_args, opf_core, opf_model
export ac_opf_recipe, ac_opf_args
export dcopf_recipe
export mpopf_recipe, mpopf_args
export mpopf_args_default, MPOPF_DEFAULT_CURVE
export scopf_recipe, scopf_args
export scopf_args_default, SCOPF_DEFAULT_CONTINGENCIES
    
# A `const Ref`, not a plain global. `global TMPDIR = ...` leaves the binding
# typed `Any`, and `mkpath(TMPDIR::Any)` is then an unresolved call that
# `juliac --trim=safe` refuses — `__init__` is part of the compiled image.
const TMPDIR = Ref{String}("")

function __init__()
    if haskey(ENV, "EXA_MODELS_DEPOT")
        TMPDIR[] = ENV["EXA_MODELS_DEPOT"]
    else
        TMPDIR[] = joinpath(@__DIR__, "..", "data")
        mkpath(TMPDIR[])
    end
    return nothing
end

end # module ExaModelsPower
