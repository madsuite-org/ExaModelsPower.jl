# Precompiles OPF model building and evaluation on NVIDIA GPUs (CUDA) (see src/precompile.jl). Needs a
# functional device at precompile time. `ExaModels` and `KernelAbstractions` are
# triggers so that ExaModels' backend extensions are loaded while this one precompiles.
module ExaModelsPowerCUDAExt

import ExaModelsPower
import CUDACore
import PrecompileTools

PrecompileTools.@compile_workload begin
    if CUDACore.functional()
        ExaModelsPower._precompile_workload(CUDACore.CUDAKernels.CUDABackend())
    end
end

end
