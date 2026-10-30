# Starting from the case's own operating point: the bus voltages (VM, VA) and the
# generator dispatch (PG, QG) that the case file gives, instead of the flat start.
# For a solved case, such as the one `write_matpower_solution` writes, that is a
# warm start from its solution.

# Named functions rather than closures, so that the start survives ahead-of-time
# compilation (see `opf_start`). The parse keeps VA in degrees, as the case file gives
# it; the model's angles are in radians.
_case_va(d) = d.va0 .* (pi / 180)
_case_vm(d) = d.vm0
_case_vr(d) = d.vm0 .* cos.(_case_va(d))
_case_vim(d) = d.vm0 .* sin.(_case_va(d))
_case_pg(d) = d.pg0
_case_qg(d) = d.qg0

"""
    CASE_START

The starting points `opf_args(file; start = :case)` uses: the case file's bus voltages
(`va`, `vm`, and `vr`, `vim` from them) and generator dispatch (`pg`, `qg`). The branch
flows keep their default.
"""
const CASE_START = (va = _case_va, vm = _case_vm, vr = _case_vr, vim = _case_vim, pg = _case_pg, qg = _case_qg)

_start_overrides(s::Symbol) =
    s === :case ? CASE_START :
    throw(ArgumentError("start must be :case, or a NamedTuple of starting points; got :$s"))
_start_overrides(s) = NamedTuple(s)

"""
    case_start(case_file, model) -> Vector

The starting point of `model` (built by [`opf_model`](@ref), in any formulation) with the
bus voltages and generator dispatch taken from `case_file`, as [`CASE_START`](@ref)
defines them: `model`'s own `x0` with those blocks replaced. `case_file` must be the case
`model` was built from, or one with the same buses and generators, such as its solution
written by [`write_matpower_solution`](@ref).
"""
function case_start(case_file::AbstractString, model::ExaModels.ExaModel)
    x = Array(model.meta.x0)
    T = eltype(x)
    d = only(opf_args(case_file, T, nothing, CASE_START))
    _place_start!(x, _mp_form(model.refs), model.refs, d)
    return x
end

function _place_start!(x, v, a)
    b = _mp_blk(v, x)
    length(b) == length(a) || throw(DimensionMismatch(
        "the case has $(length(a)) entries for a variable block of length $(length(b)): " *
        "it is not the case the model was built from"))
    copyto!(b, a)
    return x
end
_place_start!(x, ::Polar, r, d) =
    (_place_start!(x, r.va, d.va_start); _place_start!(x, r.vm, d.vm_start);
     _place_start!(x, r.pg, d.pg_start); _place_start!(x, r.qg, d.qg_start))
_place_start!(x, ::Rect, r, d) =
    (_place_start!(x, r.vr, d.vr_start); _place_start!(x, r.vim, d.vim_start);
     _place_start!(x, r.pg, d.pg_start); _place_start!(x, r.qg, d.qg_start))
_place_start!(x, ::DC, r, d) = (_place_start!(x, r.va, d.va_start); _place_start!(x, r.pg, d.pg_start))

"""
    write_case_start(out, case_file, model, x, y, zl, zu; objective = NaN, status = "")

Write [`case_start`](@ref)`(case_file, model)` to `out`, one number per line. The
signature of a solution writer, so that a compiled library carries it (`P_write_start`;
see `compile_all`): a program using the library reads the file back as the starting
point. `x`, the multipliers, `objective` and `status` are not used.
"""
function write_case_start(out::AbstractString, case_file::AbstractString, model::ExaModels.ExaModel,
                          x, y, zl, zu; objective = NaN, status::AbstractString = "")
    s = case_start(case_file, model)
    io = open(out, "w")
    try
        for v in s
            write(io, string(Float64(v)), "\n")
        end
    finally
        close(io)
    end
    return out
end
