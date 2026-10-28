# Precompiles OPF model building and evaluation on NVIDIA GPUs (CUDA) (see src/precompile.jl). Needs a
# functional device at precompile time. `ExaModels` and `KernelAbstractions` are
# triggers so that ExaModels' backend extensions are loaded while this one precompiles.
module ExaModelsPowerCUDAExt

import ExaModelsPower
import CUDA
import PrecompileTools

PrecompileTools.@compile_workload begin
    if CUDA.functional()
        ExaModelsPower._precompile_workload(CUDA.CUDABackend())
    end
end

end
