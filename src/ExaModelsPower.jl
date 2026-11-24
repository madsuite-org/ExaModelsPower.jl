module ExaModelsPower

import ExaModels: ExaModels, ExaCore, @add_var, @add_par, @add_con, @add_obj, @add_con!, ExaModel, convert_array, solution
using DelimitedFiles
using ExaPowerIO
using JSON
import PowerIO

# Parsers and data stay separate from the models that consume them.
include("parser.jl")          # matpower -> the tables every model is built from
include("constraint.jl")      # the algebraic expressions, shared by all three
include("goc3_parser.jl")     # a GOC3 problem, through PowerIO
include("sc_parser.jl")       # the GOC3 model's arrays, at a fixed commitment

include("opf.jl")             # static: polar, rect, DC
include("mpopf.jl")           # multi-period: polar, rect, DC
include("goc3.jl")            # security-constrained, GOC3 formulation
include("matpower_output.jl")  # solved MATPOWER case from an OPF solution
include("violations.jl")       # constraint and bound violations by group
include("reports.jl")          # text about an instance and a solution (compiled libraries)
include("case_start.jl")       # starting from the case's operating point
include("precompile.jl")       # precompile workloads (CPU here, GPUs in extensions)

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
export write_matpower_solution, opf_violations, write_violations
export opf_report_instance, opf_report_solution
export CASE_START, case_start, write_case_start
export opf_recipe, opf_args, opf_core, opf_model
export ac_opf_recipe, ac_opf_args
export dcopf_recipe
export mpopf_recipe, mpopf_args
export mpopf_args_default, MPOPF_DEFAULT_CURVE
    
# A `const Ref`, not a plain global: `__init__` is part of a compiled (`juliac`) image,
# where an untyped global is an unresolved call. The directory is not created here: in a
# compiled library `@__DIR__` is the path on the machine that built it, which another
# machine may not be allowed to create.
const TMPDIR = Ref{String}("")

function __init__()
    TMPDIR[] = get(ENV, "EXA_MODELS_DEPOT", joinpath(@__DIR__, "..", "data"))
    return nothing
end

end # module ExaModelsPower
