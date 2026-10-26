# Precompile workloads. Building a model and evaluating its callbacks compiles the model
# building code and, on a GPU, the model's kernels. The model type does not depend on the
# case, so the bundled case14 covers every case of the same formulation. The CPU workload
# runs here; the GPU and KernelAbstractions workloads run in the backend extensions.

import PrecompileTools
import ExaModels: NLPModels

# `meta` and the callbacks are reached through `invokelatest`, the way a solver behind a
# model barrier (MadNLP's `ModelBarrier`) reaches them, so each is compiled as a
# standalone method rather than only inlined into this function.
function _evaluate_callbacks(m)
    x = copy(NLPModels.get_x0(m))
    T = eltype(x)
    ncon, nnzj, nnzh = NLPModels.get_ncon(m), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m)
    y = fill!(similar(x, ncon), zero(T))
    call(f, args...; kw...) = Base.invokelatest(f, args...; kw...)
    call(getproperty, m, :meta)
    call(NLPModels.jac_structure!, m, similar(x, Int, nnzj), similar(x, Int, nnzj))
    call(NLPModels.hess_structure!, m, similar(x, Int, nnzh), similar(x, Int, nnzh))
    call(NLPModels.obj, m, x)
    call(NLPModels.grad!, m, x, similar(x))
    call(NLPModels.cons!, m, x, similar(x, ncon))
    call(NLPModels.jac_coord!, m, x, similar(x, nnzj))
    call(NLPModels.hess_coord!, m, x, y, similar(x, nnzh); obj_weight = one(T))
    call(NLPModels.jprod!, m, x, similar(x), similar(x, ncon))
    call(NLPModels.jtprod!, m, x, similar(x, ncon), similar(x))
    # MadNLP passes a view of a longer buffer when the model has no fixed variables
    H = similar(x, nnzh + 1)
    call(NLPModels.hess_coord!, m, x, y, view(H, 1:nnzh); obj_weight = one(T))
    J = similar(x, nnzj + 1)
    call(NLPModels.jac_coord!, m, x, view(J, 1:nnzj))
    return nothing
end

_workload_case() = joinpath(pkgdir(@__MODULE__), "data", "pglib_opf_case14_ieee.m")

# Write `copies` disconnected copies of `case` into one MATPOWER file at `path`, with the
# bus numbers of copy `c` offset by `c * 1000`. Built on a GPU, the larger model sorts its
# sparsity patterns over more than one thread block, which runs kernel launches that
# case14 alone does not.
function _replicate_case(case, path, copies)
    cols = Dict("bus" => (1,), "gen" => (1,), "branch" => (1, 2), "gencost" => ())
    section, rows = nothing, String[]
    open(path, "w") do io
        for line in eachline(case)
            s = strip(line)
            m = match(r"^mpc\.(\w+)\s*=\s*\[", s)
            if m !== nothing && haskey(cols, m[1])
                section, rows = m[1], String[]
            elseif section !== nothing && startswith(s, "];")
                for c in 0:(copies-1), row in rows
                    fields = split(first(split(row, ';')))
                    for j in cols[section]
                        fields[j] = string(parse(Int, fields[j]) + 1000c)
                    end
                    println(io, "\t", join(fields, "\t"), ";")
                end
                section = nothing
            elseif section !== nothing
                (isempty(s) || startswith(s, '%')) || push!(rows, s)
                continue
            end
            println(io, line)
        end
    end
    return path
end

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
    # Called through `invokelatest`, as from the top level, so the keyword entry points
    # are compiled as standalone methods rather than only inlined into this function.
    build(file; kws...) = first(Base.invokelatest(ac_opf_model, file; kws...))
    m = build(case; kw...)                             # default form, no `form` keyword
    _evaluate_callbacks(m)
    backend === nothing && build(case; backend)        # an explicit `backend = nothing`
    for form in (Polar(), Rect(), DC())
        m = build(case; kw..., form)
        _evaluate_callbacks(m)
    end
    if backend !== nothing
        mktempdir() do dir
            large = _replicate_case(case, joinpath(dir, "case14x8.m"), 8)
            for form in (Polar(), Rect(), DC())
                _evaluate_callbacks(build(large; kw..., form))
            end
        end
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
    # Resolving a case by name (`ac_opf_model("pglib_opf_case118_ieee.m")`) looks up
    # ExaPowerIO's lazy PGLib artifact. Compile that lookup without running it, since
    # running it may download the artifact.
    LA = ExaPowerIO.LazyArtifacts
    for f in (LA.Artifacts.__artifact_str, LA.Artifacts._artifact_str)
        precompile(f, (Module, String, SubString{String}, String, Dict{String,Any},
                       Base.SHA1, Base.BinaryPlatforms.Platform, Val{LA}))
    end
end
