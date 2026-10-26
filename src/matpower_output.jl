# Write an OPF solution as a solved MATPOWER case: the input case file with the
# bus, gen, and branch tables updated and the result columns appended, in the
# layout MATPOWER's `savecase` uses for an OPF result.

"""
    write_matpower_solution(out, case_file, vars, cons, x, y, zl, zu;
                            form = Polar(), objective = NaN, status = "")

Write the OPF solution `x` of the model `opf_model(case_file; form)` as a
solved MATPOWER case to `out` (a path or an `IO`), for `form = Polar()`,
`Rect()` or `DC()`.

`vars` and `cons` are the variable and constraint handles `opf_model` returns,
`y` the constraint multipliers, and `zl`, `zu` the lower and upper bound
multipliers, with Ipopt's signs (`L = f + yᵀg`, `zl, zu ≥ 0`).

The output is `case_file` with, in MATPOWER's units and column order:

- `bus`: `VM`, `VA` (degrees) updated; `LAM_P`, `LAM_Q` (\$/MWh, \$/MVArh per hour of
  objective), `MU_VMAX`, `MU_VMIN` (per p.u.) appended;
- `gen`: `PG`, `QG` (MW, MVAr) and `VG` updated; `MU_PMAX`, `MU_PMIN`, `MU_QMAX`,
  `MU_QMIN` (per MW, per MVAr) appended;
- `branch`: `PF`, `QF`, `PT`, `QT` (MW, MVAr), `MU_SF`, `MU_ST` (per MVA),
  `MU_ANGMIN`, `MU_ANGMAX` (per degree) appended.

For `DC()`, as in MATPOWER's DC OPF, `VM` is 1, the reactive quantities and
their multipliers are 0, `PT = -PF`, and `MU_SF`, `MU_ST` are the multipliers of
the flow limit in the from-to and to-from direction.

Out-of-service generators and branches get zeros, as in MATPOWER. Everything
else in the file is kept as it was, and `objective` and `status` go into a
comment at the top.
"""
function write_matpower_solution(out::AbstractString, case_file::AbstractString, vars, cons, x, y, zl, zu; kwargs...)
    # Not `open(f, out, "w")`: it splats its arguments, which `--trim=safe` refuses.
    io = open(out, "w")
    try
        write_matpower_solution(io, case_file, vars, cons, x, y, zl, zu; kwargs...)
    finally
        close(io)
    end
    return out
end

"""
    write_matpower_solution(out, case_file, model, x, y, zl, zu; objective = NaN, status = "")

The same, for a `model` built by `opf_model`: the variable and constraint
handles and the formulation are read from the model. This is the form a
compiled library carries; see `compile_all`.
"""
function write_matpower_solution(out::AbstractString, case_file::AbstractString, model::ExaModels.ExaModel,
                                 x, y, zl, zu; objective = NaN, status::AbstractString = "")
    r = model.refs
    return write_matpower_solution(out, case_file, r, r, x, y, zl, zu; form = _mp_form(r), objective, status)
end

# The formulation a model was built with, by dispatch on its voltage block:
# `vm` (polar), `vr` (rectangular), or neither (DC).
_mp_form(r::NamedTuple) = _mp_form(Val(hasfield(typeof(r), :vm)), Val(hasfield(typeof(r), :vr)))
_mp_form(::Val{true}, ::Val{false}) = Polar()
_mp_form(::Val{false}, ::Val{true}) = Rect()
_mp_form(::Val{false}, ::Val{false}) = DC()

function write_matpower_solution(
    io::IO, case_file::AbstractString, vars, cons, x, y, zl, zu;
    form::OPFForm = Polar(), objective = NaN, status::AbstractString = "",
)
    lines = _mp_readlines(_case_path(case_file))
    base = _mp_base_mva(lines)
    s = _mp_columns(form, vars, cons, x, y, zl, zu, base)
    degree = π / 180

    println(io, "%% Solved by ExaModelsPower: objective = ", objective, ", status = ", status)
    println(io, "%% Result columns in MATPOWER's savecase layout and units.")
    table = ""
    k = 0                      # active rows seen in the current table
    n = 0                      # rows of the solution for the current table
    tables = 0
    bus_vm = Dict{Int,Float64}()
    for line in lines
        if isempty(table)
            m = match(r"^\s*mpc\.(bus|gen|branch)\s*=\s*\[", line)
            if m !== nothing
                table = m.captures[1]
                k = 0
                n = _mp_rows(table, s)
                tables += 1
            end
            println(io, line)
            continue
        end
        code, comment = _mp_split_comment(line)
        if occursin(']', code)
            k == n || throw(_mp_mismatch(table, n, k))
            table = ""
            println(io, line)
            continue
        end
        row = split(strip(replace(code, ';' => ' ', ',' => ' ')))
        if isempty(row)
            println(io, line)
            continue
        end
        # The input's own columns only, so a case that already holds results is
        # rewritten rather than extended twice.
        ncol = table == "gen" ? 21 : 13
        out = String.(row[1:min(end, ncol)])
        if table == "bus"
            # BUS_TYPE 4: isolated, not in the model.
            if parse(Int, out[2]) != 4
                k = _mp_next(k, n, table)
                out[8] = _mp_num(s.vm[k]); out[9] = _mp_num(s.va[k] / degree)
                bus_vm[parse(Int, out[1])] = s.vm[k]
                append!(out, _mp_num.((s.lam_p[k], s.lam_q[k], s.mu_vmax[k], s.mu_vmin[k])))
            else
                append!(out, ("0", "0", "0", "0"))
            end
        elseif table == "gen"
            if parse(Float64, out[8]) != 0
                k = _mp_next(k, n, table)
                out[2] = _mp_num(s.pg[k] * base); out[3] = _mp_num(s.qg[k] * base)
                out[6] = _mp_num(get(bus_vm, parse(Int, out[1]), parse(Float64, out[6])))
                _mp_pad!(out, 21)
                append!(out, _mp_num.((s.mu_pmax[k], s.mu_pmin[k], s.mu_qmax[k], s.mu_qmin[k])))
            else
                out[2] = "0"; out[3] = "0"
                _mp_pad!(out, 21)
                append!(out, ("0", "0", "0", "0"))
            end
        elseif table == "branch"
            _mp_pad!(out, 13)
            if parse(Float64, out[11]) != 0
                k = _mp_next(k, n, table)
                append!(out, _mp_num.((s.pf[k] * base, s.qf[k] * base, s.pt[k] * base, s.qt[k] * base)))
                append!(out, _mp_num.((s.mu_sf[k], s.mu_st[k], s.mu_angmin[k], s.mu_angmax[k])))
            else
                append!(out, ("0", "0", "0", "0", "0", "0", "0", "0"))
            end
        end
        println(io, "\t", join(out, '\t'), ";", isempty(comment) ? "" : "\t" * comment)
    end
    tables == 3 || throw(ArgumentError(case_file * " is not a MATPOWER case: the solution " *
                                       "is written into the case file's own bus, gen and branch tables"))
    return io
end

# Each in-service row of a table takes the next entry of the solution, so the
# two must agree in number. A solution of another case, or of a network edited
# after it was read, is refused rather than written into the wrong rows.
_mp_rows(table, s) = table == "bus" ? length(s.vm) : table == "gen" ? length(s.pg) : length(s.pf)
_mp_next(k, n, table) = k < n ? k + 1 : throw(_mp_mismatch(table, n, k + 1))
_mp_mismatch(table, n, k) = DimensionMismatch(
    "the solution has " * string(n) * " " * table * " rows but the case file has " *
    (k > n ? "more than " * string(n) : string(k)) * " in service")

_mp_blk(v, a) = view(a, (v.offset + 1):(v.offset + ExaModels.total(v.size)))

# The result columns of a solution, per active bus, generator and branch, in the
# units MATPOWER reports them in, except powers (p.u., scaled when written) and
# angles (radians).
function _mp_columns(form::Union{Polar,Rect}, vars, cons, x, y, zl, zu, base)
    blk = _mp_blk
    vm, va, mu_vmax, mu_vmin = _mp_voltages(form, vars, cons, x, y, zl, zu)
    lam_p = blk(cons.c_active_power_balance, y) ./ base
    lam_q = blk(cons.c_reactive_power_balance, y) ./ base

    # Arc l is the from end of branch l, arc l + nbranch the to end.
    p, q = blk(vars.p, x), blk(vars.q, x)
    nbr = length(p) ÷ 2
    f, t = 1:nbr, (nbr + 1):(2nbr)
    # p² + q² ≤ rate²: d/drate of the squared limit is 2 rate, and where the
    # multiplier is nonzero the limit is active, so rate = |(p, q)|. That holds
    # for a limit the parser derived (MATPOWER's rateA = 0) as well.
    lsf, lst = blk(cons.c_from_thermal_limit, y), blk(cons.c_to_thermal_limit, y)
    lang = blk(cons.c_phase_angle_diff, y)
    degree = π / 180
    return (;
        vm, va, lam_p, lam_q, mu_vmax, mu_vmin,
        pg = collect(blk(vars.pg, x)), qg = collect(blk(vars.qg, x)),
        mu_pmax = blk(vars.pg, zu) ./ base, mu_pmin = blk(vars.pg, zl) ./ base,
        mu_qmax = blk(vars.qg, zu) ./ base, mu_qmin = blk(vars.qg, zl) ./ base,
        pf = p[f], qf = q[f], pt = p[t], qt = q[t],
        mu_sf = max.(lsf, 0) .* 2 .* hypot.(p[f], q[f]) ./ base,
        mu_st = max.(lst, 0) .* 2 .* hypot.(p[t], q[t]) ./ base,
        mu_angmin = max.(.-lang, 0) .* degree, mu_angmax = max.(lang, 0) .* degree,
    )
end

# Voltage magnitude and angle per bus, and the multipliers of the magnitude bounds.
function _mp_voltages(::Polar, vars, cons, x, y, zl, zu)
    blk = _mp_blk
    return collect(blk(vars.vm, x)), collect(blk(vars.va, x)),
           collect(blk(vars.vm, zu)), collect(blk(vars.vm, zl))
end

function _mp_voltages(::Rect, vars, cons, x, y, zl, zu)
    blk = _mp_blk
    vr, vim = blk(vars.vr, x), blk(vars.vim, x)
    vm = hypot.(vr, vim)
    # vmin² ≤ vr² + vim² ≤ vmax²: d/dvm of the squared bound is 2 vm.
    lv = blk(cons.c_voltage_magnitude, y)
    return vm, atan.(vim, vr), max.(lv, 0) .* 2 .* vm, max.(.-lv, 0) .* 2 .* vm
end

function _mp_columns(::DC, vars, cons, x, y, zl, zu, base)
    blk = _mp_blk
    va = collect(blk(vars.va, x))
    pg = collect(blk(vars.pg, x))
    pf = collect(blk(vars.pf, x))
    lang = blk(cons.c_phase_angle_diff, y)
    degree = π / 180
    return (;
        vm = ones(length(va)), va,
        lam_p = blk(cons.c_active_power_balance, y) ./ base,
        lam_q = zeros(length(va)), mu_vmax = zeros(length(va)), mu_vmin = zeros(length(va)),
        pg, qg = zeros(length(pg)),
        mu_pmax = blk(vars.pg, zu) ./ base, mu_pmin = blk(vars.pg, zl) ./ base,
        mu_qmax = zeros(length(pg)), mu_qmin = zeros(length(pg)),
        # -rate ≤ pf ≤ rate: the upper bound limits the from-to flow.
        pf, qf = zeros(length(pf)), pt = .-pf, qt = zeros(length(pf)),
        mu_sf = blk(vars.pf, zu) ./ base, mu_st = blk(vars.pf, zl) ./ base,
        mu_angmin = max.(.-lang, 0) .* degree, mu_angmax = max.(lang, 0) .* degree,
    )
end

# `readlines` without `eachline`, whose `ondone` closure `--trim=safe` cannot resolve.
function _mp_readlines(path)
    io = open(path, "r")
    text = try
        read(io, String)
    finally
        close(io)
    end
    lines = [chopsuffix(l, "\r") for l in split(text, '\n')]
    isempty(lines) || isempty(last(lines)) && pop!(lines)
    return lines
end

# mpc.baseMVA = 100;
function _mp_base_mva(lines)
    for line in lines
        m = match(r"^\s*mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", line)
        m === nothing || return parse(Float64, m.captures[1])
    end
    return 100.0
end

function _mp_split_comment(line)
    i = findfirst('%', line)
    return i === nothing ? (line, "") : (line[1:prevind(line, i)], line[i:end])
end

_mp_num(v::Real) = isfinite(v) ? string(Float64(v)) : "0"

# Optional columns MATPOWER defines but a case file may leave out.
function _mp_pad!(row, n)
    while length(row) < n
        push!(row, "0")
    end
    return row
end
