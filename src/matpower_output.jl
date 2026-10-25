# Write an AC OPF solution as a solved MATPOWER case: the input case file with
# the bus, gen, and branch tables updated and the result columns appended, in
# the layout MATPOWER's `savecase` uses for an OPF result.

"""
    write_matpower_solution(out, case_file, vars, cons, x, y, zl, zu;
                            form = Polar(), objective = NaN, status = "")

Write the AC OPF solution `x` of the model `opf_model(case_file; form)` as a
solved MATPOWER case to `out` (a path or an `IO`).

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

The same, for a `model` built by `opf_model` (polar or rectangular): the
variable and constraint handles and the formulation are read from the model.
This is the form a compiled library carries; see `compile_all`.
"""
function write_matpower_solution(out::AbstractString, case_file::AbstractString, model::ExaModels.ExaModel,
                                 x, y, zl, zu; objective = NaN, status::AbstractString = "")
    r = model.refs
    form = hasproperty(r, :vm) ? Polar() : Rect()
    return write_matpower_solution(out, case_file, r, r, x, y, zl, zu; form, objective, status)
end

function write_matpower_solution(
    io::IO, case_file::AbstractString, vars, cons, x, y, zl, zu;
    form::Union{Polar,Rect} = Polar(), objective = NaN, status::AbstractString = "",
)
    lines = _mp_readlines(_mp_case_path(case_file))
    base = _mp_base_mva(lines)
    blk(v, a) = view(a, (v.offset + 1):(v.offset + ExaModels.total(v.size)))
    polar = form isa Polar

    # Per active bus, in file order (the order the model's buses are in).
    a, b = polar ? (blk(vars.va, x), blk(vars.vm, x)) : (blk(vars.vr, x), blk(vars.vim, x))
    vm = polar ? collect(b) : hypot.(a, b)
    va = polar ? collect(a) : atan.(b, a)
    lam_p = blk(cons.c_active_power_balance, y) ./ base
    lam_q = blk(cons.c_reactive_power_balance, y) ./ base
    if polar
        mu_vmax = collect(blk(vars.vm, zu))
        mu_vmin = collect(blk(vars.vm, zl))
    else
        # vmin² ≤ vr² + vim² ≤ vmax²: d/dvm of the squared bound is 2 vm.
        lv = blk(cons.c_voltage_magnitude, y)
        mu_vmax = max.(lv, 0) .* 2 .* vm
        mu_vmin = max.(.-lv, 0) .* 2 .* vm
    end

    # Per active generator.
    pg, qg = blk(vars.pg, x), blk(vars.qg, x)
    mu_pmax, mu_pmin = blk(vars.pg, zu) ./ base, blk(vars.pg, zl) ./ base
    mu_qmax, mu_qmin = blk(vars.qg, zu) ./ base, blk(vars.qg, zl) ./ base

    # Per active branch: arc l is the from end of branch l, arc l + nbranch the to end.
    p, q = blk(vars.p, x), blk(vars.q, x)
    nbr = length(p) ÷ 2
    # p² + q² ≤ rate²: d/drate of the squared limit is 2 rate (both in p.u.).
    lsf, lst = blk(cons.c_from_thermal_limit, y), blk(cons.c_to_thermal_limit, y)
    lang = blk(cons.c_phase_angle_diff, y)
    degree = π / 180

    println(io, "%% Solved by ExaModelsPower: objective = ", objective, ", status = ", status)
    println(io, "%% Result columns in MATPOWER's savecase layout and units.")
    table = ""
    k = 0                      # active rows seen in the current table
    rates = Float64[]          # rate_a (p.u.) of the active branches, from the file
    bus_vm = Dict{Int,Float64}()
    for line in lines
        if isempty(table)
            m = match(r"^\s*mpc\.(bus|gen|branch)\s*=\s*\[", line)
            if m !== nothing
                table = m.captures[1]
                k = 0
            end
            println(io, line)
            continue
        end
        code, comment = _mp_split_comment(line)
        if occursin(']', code)
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
                k += 1
                out[8] = _mp_num(vm[k]); out[9] = _mp_num(va[k] / degree)
                bus_vm[parse(Int, out[1])] = vm[k]
                append!(out, _mp_num.((lam_p[k], lam_q[k], mu_vmax[k], mu_vmin[k])))
            else
                append!(out, ("0", "0", "0", "0"))
            end
        elseif table == "gen"
            if parse(Float64, out[8]) != 0
                k += 1
                out[2] = _mp_num(pg[k] * base); out[3] = _mp_num(qg[k] * base)
                out[6] = _mp_num(get(bus_vm, parse(Int, out[1]), parse(Float64, out[6])))
                _mp_pad!(out, 21)
                append!(out, _mp_num.((mu_pmax[k], mu_pmin[k], mu_qmax[k], mu_qmin[k])))
            else
                out[2] = "0"; out[3] = "0"
                _mp_pad!(out, 21)
                append!(out, ("0", "0", "0", "0"))
            end
        elseif table == "branch"
            _mp_pad!(out, 13)
            if parse(Float64, out[11]) != 0
                k += 1
                rate = parse(Float64, out[6]) / base
                f, t = k, k + nbr
                append!(out, _mp_num.((p[f] * base, q[f] * base, p[t] * base, q[t] * base)))
                append!(out, _mp_num.((
                    max(lsf[k], 0) * 2 * rate / base, max(lst[k], 0) * 2 * rate / base,
                    max(-lang[k], 0) * degree, max(lang[k], 0) * degree,
                )))
            else
                append!(out, ("0", "0", "0", "0", "0", "0", "0", "0"))
            end
        end
        println(io, "\t", join(out, '\t'), ";", isempty(comment) ? "" : "\t" * comment)
    end
    return io
end

# A name that is not a file is a PGLib case, as `opf_args` reads it.
_mp_case_path(f) = isfile(f) ? f : joinpath(ExaPowerIO.get_path(:pglib), f)

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
