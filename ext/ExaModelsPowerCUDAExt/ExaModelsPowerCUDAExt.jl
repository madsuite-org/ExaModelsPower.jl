# Precompiles the GPU automatic differentiation of the OPF models. Building a model and
# evaluating its callbacks on `CUDABackend` compiles the model's kernels; the model type
# does not depend on the case, so one small case covers every case of the same form.
# Needs a functional GPU at precompile time; the cached cubins target that GPU.
# `ExaModels` is listed as a trigger so that ExaModels' KernelAbstractions extension,
# which provides the GPU model path, is loaded while this extension precompiles.
module ExaModelsPowerCUDAExt

import ExaModelsPower
import ExaModels
import ExaModels: NLPModels
import CUDACore
import KernelAbstractions
import PrecompileTools

function evaluate_callbacks(m)
    x = copy(NLPModels.get_x0(m))
    y = fill!(similar(x, NLPModels.get_ncon(m)), zero(eltype(x)))
    g = similar(x)
    c = similar(x, NLPModels.get_ncon(m))
    jrows = similar(x, Int, NLPModels.get_nnzj(m)); jcols = similar(jrows)
    hrows = similar(x, Int, NLPModels.get_nnzh(m)); hcols = similar(hrows)
    jac = similar(x, NLPModels.get_nnzj(m))
    hess = similar(x, NLPModels.get_nnzh(m))
    NLPModels.jac_structure!(m, jrows, jcols)
    NLPModels.hess_structure!(m, hrows, hcols)
    NLPModels.obj(m, x)
    NLPModels.grad!(m, x, g)
    NLPModels.cons!(m, x, c)
    NLPModels.jac_coord!(m, x, jac)
    NLPModels.hess_coord!(m, x, y, hess; obj_weight = one(eltype(x)))
    CUDACore.synchronize()
    return nothing
end

PrecompileTools.@setup_workload begin
    if CUDACore.functional()
        case = joinpath(pkgdir(ExaModelsPower), "data", "pglib_opf_case14_ieee.m")
        backend = CUDACore.CUDAKernels.CUDABackend()
        PrecompileTools.@compile_workload begin
            # the default form is also called without `form`, which is a separate keyword
            # call specialization
            m, _, _ = ExaModelsPower.ac_opf_model(case; backend)
            evaluate_callbacks(m)
            for form in (ExaModelsPower.Polar(), ExaModelsPower.Rect(), ExaModelsPower.DC())
                m, _, _ = ExaModelsPower.ac_opf_model(case; backend, form)
                evaluate_callbacks(m)
            end
            # Large cases reach kernel variants that case14 does not. `obj` sums the
            # objective terms, and above one thread block CUDA.jl reduces with a
            # multi-block kernel. Model construction runs `findall` over the sparsity
            # patterns, and above ~10^6 entries its scan nests a second multi-block scan.
            sum(CUDACore.zeros(Float64, 1 << 16))
            findall(CUDACore.fill(true, 1 << 21))
        end
    end
end

end
