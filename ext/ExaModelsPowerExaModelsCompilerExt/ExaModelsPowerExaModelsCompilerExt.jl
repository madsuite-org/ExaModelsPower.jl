module ExaModelsPowerExaModelsCompilerExt

# ExaModelsPower's side of the `compile_all` contract. It lives in an extension
# so the package never depends on the compiler: `using ExaModelsCompiler`
# activates this, and nothing changes for anyone who only builds models.
#
# The shape this package needs, and why it differs from a size-parameterized
# provider: an OPF model is not instantiated at a SIZE, it is instantiated at a
# CASE. Every model here carries an argument function — `opf_args` for the
# static formulations, `mpopf_args_default` for multi-period — which the
# compiled library calls with a matpower path, so what crosses the C boundary
# is a string rather than an integer. `sizes` has nothing to mean here;
# `cases` does.

using ExaModelsPower
using ExaModelsCompiler

# The models this package can compile, each with the instantiation spec it
# needs. Multi-period bakes (N, curve, Nbus) into its recipe, so its entry is
# per-case in a way the static ones are not — the spec belongs to the model,
# not to the call.
function _emp_models(case::AbstractString; T = Float64, N = length(ExaModelsPower.MPOPF_DEFAULT_CURVE))
    static(form) = (ExaModelsPower.opf_recipe(; form = form, T = T)[1],
                    ExaModelsPower.opf_args, case)
    multi(form) = (ExaModelsPower.mpopf_recipe(; N = N, form = form, T = T)[1],
                   ExaModelsPower.mpopf_args_default, case)
    # `K` is not a knob the caller gets, unlike `N`. The compiled library calls
    # the argument function with a path and nothing else, so the contingency
    # list is whatever `scopf_args_default` holds -- and the recipe must bake
    # the same K, or the bounds and the structure disagree. Deriving it from
    # that list is what keeps the two from drifting apart.
    scopf(form) = (ExaModelsPower.scopf_recipe(;
                       K = length(ExaModelsPower.SCOPF_DEFAULT_CONTINGENCIES),
                       form = form, T = T)[1],
                   ExaModelsPower.scopf_args_default, case)
    return [
        :acp   => static(ExaModelsPower.Polar()),
        :acr   => static(ExaModelsPower.Rect()),
        :dcp   => static(ExaModelsPower.DC()),
        :mpacp => multi(ExaModelsPower.Polar()),
        :mpacr => multi(ExaModelsPower.Rect()),
        :mpdcp => multi(ExaModelsPower.DC()),
        :scacp => scopf(ExaModelsPower.Polar()),
        :scacr => scopf(ExaModelsPower.Rect()),
        :scdcp => scopf(ExaModelsPower.DC()),
    ]
end

"""
    compile_all(ExaModelsPower; path = "@emp", case, T, N, only, exclude, kwargs...)

Compile this package's models into one shared library.

`case` is the matpower file the recipes are built against — its TYPES are what
the compiler needs, not its values, so the resulting library instantiates any
case at run time from a path. The multi-period models additionally bake `N` and the
load curve, which is why they are per-`N` in a way the static ones are not; the
security-constrained models bake the contingency list the same way.

`path` defaults to the shared depot entry `"@emp"`.

```julia
compile_all(ExaModelsPower)                        # all nine, into "@emp"
compile_all(ExaModelsPower; only = [:acp, :dcp])
```
"""
function ExaModelsCompiler.compile_all(
    ::Val{ExaModelsPower};
    path = "@emp",
    case::AbstractString = "pglib_opf_case14_ieee.m",
    T::Type = Float64,
    N::Integer = length(ExaModelsPower.MPOPF_DEFAULT_CURVE),
    only = nothing,
    exclude = (),
    kwargs...,
)
    models = _emp_models(case; T = T, N = N)
    chosen = ExaModelsCompiler.select(models; only = only, exclude = exclude)
    return ExaModelsCompiler.compile_library(path, chosen...; kwargs...)
end

end # module
