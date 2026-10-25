# Precompiles OPF model building and evaluation on Apple GPUs (Metal), in Float32 (see
# src/precompile.jl). Needs a functional device at precompile time. `ExaModels` and `KernelAbstractions` are
# triggers so that ExaModels' backend extensions are loaded while this one precompiles.
module ExaModelsPowerMetalExt

import ExaModelsPower
import Metal
import PrecompileTools

PrecompileTools.@compile_workload begin
    if Metal.functional()
        ExaModelsPower._precompile_workload(Metal.MetalBackend(); T = Float32)
    end
end

end
