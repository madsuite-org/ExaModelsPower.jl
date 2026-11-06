# Precompiles OPF model building and evaluation on the KernelAbstractions CPU backend (see
# src/precompile.jl). `ExaModels` is a trigger so that ExaModels' KernelAbstractions
# extension is loaded while this one precompiles. `LinearAlgebra` and `SparseArrays` are
# triggers so that KernelAbstractions' extensions for them are too: they add `get_backend`
# methods, and code stored without them is rejected when this image loads, since both are
# always loaded by then.
module ExaModelsPowerKernelAbstractionsExt

import ExaModelsPower
import KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    ExaModelsPower._precompile_workload(KernelAbstractions.CPU())
    # Loading KernelAbstractions invalidates CPU code in the package's own image; this image
    # loads after those methods, so what it stores stays valid.
    ExaModelsPower._precompile_workload(nothing)
end

end
