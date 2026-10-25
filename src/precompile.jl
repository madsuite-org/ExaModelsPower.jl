# Precompile workloads. Building a model and evaluating its callbacks compiles the model
# building code and, on a GPU, the model's kernels. The model type does not depend on the
# case, so the bundled case14 covers every case of the same formulation. The CPU workload
# runs here; the GPU and KernelAbstractions workloads run in the backend extensions.

import PrecompileTools
import ExaModels: NLPModels

function _evaluate_callbacks(m)
    x = copy(NLPModels.get_x0(m))
    T = eltype(x)
    ncon, nnzj, nnzh = NLPModels.get_ncon(m), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m)
    y = fill!(similar(x, ncon), zero(T))
    NLPModels.jac_structure!(m, similar(x, Int, nnzj), similar(x, Int, nnzj))
    NLPModels.hess_structure!(m, similar(x, Int, nnzh), similar(x, Int, nnzh))
    NLPModels.obj(m, x)
    NLPModels.grad!(m, x, similar(x))
    NLPModels.cons!(m, x, similar(x, ncon))
    NLPModels.jac_coord!(m, x, similar(x, nnzj))
    NLPModels.hess_coord!(m, x, y, similar(x, nnzh); obj_weight = one(T))
    return nothing
end

_workload_case() = joinpath(pkgdir(@__MODULE__), "data", "pglib_opf_case14_ieee.m")

"""
    _precompile_workload(backend; T = nothing)

Build and evaluate the OPF models on `backend`, with the keyword combinations that
`ac_opf_model` is usually called with. `T` is passed only when given (Metal needs
`Float32`).
"""
function _precompile_workload(backend; T = nothing)
    case = _workload_case()
    kw = backend === nothing ? (;) : (; backend)
    kw = T === nothing ? kw : (; kw..., T)
    m, _, _ = ac_opf_model(case; kw...)                # default form, no `form` keyword
    _evaluate_callbacks(m)
    for form in (Polar(), Rect(), DC())
        m, _, _ = ac_opf_model(case; kw..., form)
        _evaluate_callbacks(m)
    end
    if backend !== nothing
        # Large cases reach kernel variants that case14 does not: `obj` sums the objective
        # terms, multi-block above one thread block, and model construction runs
        # `findall` over the sparsity patterns, whose scan nests a second multi-block scan
        # above ~10^6 entries.
        x = NLPModels.get_x0(m)
        sum(fill!(similar(x, 1 << 16), zero(eltype(x))))
        findall(fill!(similar(x, Bool, 1 << 21), true))
    end
    return nothing
end

PrecompileTools.@compile_workload begin
    _precompile_workload(nothing)
end
