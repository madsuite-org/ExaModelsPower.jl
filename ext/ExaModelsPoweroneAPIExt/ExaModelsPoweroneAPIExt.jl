# Precompiles OPF model building and evaluation on Intel GPUs (oneAPI) (see src/precompile.jl). Needs a
# functional device at precompile time. `ExaModels` and `KernelAbstractions` are
# triggers so that ExaModels' backend extensions are loaded while this one precompiles.
module ExaModelsPoweroneAPIExt

import ExaModelsPower
import oneAPI
import PrecompileTools

PrecompileTools.@compile_workload begin
    if oneAPI.functional()
        ExaModelsPower._precompile_workload(oneAPI.oneAPIBackend())
    end
end

end
