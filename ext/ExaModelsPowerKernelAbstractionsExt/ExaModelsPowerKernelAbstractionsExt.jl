# Precompiles OPF model building and evaluation on the KernelAbstractions CPU backend (see
# src/precompile.jl). `ExaModels` is a trigger so that ExaModels' KernelAbstractions
# extension is loaded while this one precompiles.
module ExaModelsPowerKernelAbstractionsExt

import ExaModelsPower
import KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    ExaModelsPower._precompile_workload(KernelAbstractions.CPU())
end

end
