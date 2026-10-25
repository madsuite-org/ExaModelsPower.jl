# Precompiles OPF model building and evaluation on AMD GPUs (AMDGPU) (see src/precompile.jl). Needs a
# functional device at precompile time. `ExaModels` and `KernelAbstractions` are
# triggers so that ExaModels' backend extensions are loaded while this one precompiles.
module ExaModelsPowerAMDGPUExt

import ExaModelsPower
import AMDGPU
import PrecompileTools

PrecompileTools.@compile_workload begin
    if AMDGPU.functional()
        ExaModelsPower._precompile_workload(AMDGPU.ROCBackend())
    end
end

end
