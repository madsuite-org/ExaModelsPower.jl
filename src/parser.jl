# `map`, not `NamedTuple{names}(generator)`. The generator form builds the
# tuple from an iterator whose element types inference cannot pin one by one,
# so the result is a `NamedTuple{names}` with no value types — concrete names,
# abstract contents. `map` over a NamedTuple is inferable elementwise.
# Recurses into a nested NamedTuple. The multi-period arguments group their
# N-expanded bound matrices under one field, because there are enough of them
# to push a flat tuple past the 31-field limit above.
convert_data(data::NamedTuple, backend) = map(d -> _convert_field(d, backend), data)
_convert_field(d, backend) = convert_array(d, backend)
_convert_field(d::NamedTuple, backend) = convert_data(d, backend)

# ── The rows every model reads ───────────────────────────────────────────────
#
# ExaPowerIO's row structs, whichever reader the case came through: a MATPOWER
# file read by ExaPowerIO then needs no copy of its rows, and the model bodies
# were written against these field names. All are isbits, which is what a GPU
# backend's `convert_array` and `juliac --trim=safe` need.
const BusRow{T} = ExaPowerIO.BusData{T}
const GenRow{T} = ExaPowerIO.GenData{T}
const BranchRow{T} = ExaPowerIO.BranchData{T}
const ArcRow{T} = ExaPowerIO.ArcData{T}
const StorageRow{T} = ExaPowerIO.StorageData{T}

# A row of type `R` from a NamedTuple that has its fields (and possibly more).
_row(::Type{R}, nt::NamedTuple) where {R} = R(values(NamedTuple{fieldnames(R)}(nt))...)
# A struct's fields as a NamedTuple.
_fields(x) = NamedTuple{fieldnames(typeof(x))}(ntuple(i -> getfield(x, i), Val(fieldcount(typeof(x)))))
# `x` with the fields in `nt` replaced.
_with(x, nt::NamedTuple) = _row(typeof(x), merge(_fields(x), nt))

# ── Where a case comes from ──────────────────────────────────────────────────

# A case is a MATPOWER file or another file PowerIO can read (PSS/E, PowerModels
# JSON, …), or a bare PGLib-OPF name such as "pglib_opf_case118_ieee.m", which is
# looked up in the lazily downloaded PGLib-OPF v23.07 artifact. Anything else is
# refused here rather than sent to the artifact, so a mistyped path never triggers
# the download or reports a path the caller did not write.
function _case_path(case::AbstractString)
    path = String(case)
    ispath(path) && return path
    if !isabspath(path) && startswith(basename(path), "pglib_opf_")
        return joinpath(artifact"PGLib_opf", "pglib-opf-23.07", path)
    end
    throw(ArgumentError("no case file at \"" * path * "\" (a bare name is looked up in " *
                        "PGLib-OPF only when it starts with \"pglib_opf_\")"))
end

# A MATPOWER file is read with ExaPowerIO, which parses it faster than PowerIO
# (1.7x to 11x on the 66 PGLib-OPF cases and an 82k-bus case). Every other format
# PowerIO reads goes through PowerIO, and so does a network PowerIO already parsed:
# one parsed with a `format`, or edited with `PowerIO.apply_updates!`, builds the
# same model as its file.
_is_matpower(path::AbstractString) = lowercase(splitext(path)[2]) == ".m"

parse_ac_power_data(case) = parse_ac_power_data(case, Float64)
# `T` is a TYPE PARAMETER, not a value of type `Type`. As a plain positional
# argument it arrives typed `Type` — abstract — and nothing downstream can
# specialize on it, so every field read off the result widens. Written
# `::Type{T}` it is a static parameter and the whole chain stays concrete.
function parse_ac_power_data(case::AbstractString, ::Type{T}) where {T}
    path = _case_path(case)
    # Both readers give the same row types, so the choice made at run time leaves
    # the result inferable.
    return _is_matpower(path) ? _matpower_tables(path, T) :
           _tables(_powerio_rows(PowerIO.parse(PowerIO.BalancedNetwork, path), T), T)
end

# ExaPowerIO's own filtering (`filtered = true`) drops isolated buses and
# out-of-service generators and branches as it reads, and gives the rows a model
# uses with no further copy: this is the path almost every MATPOWER case takes. It
# cannot report an element at an isolated bus (it gives it bus 0, or fails on the
# lookup), refuse a piecewise-linear cost, or filter and renumber storage. A case
# with any of these is read again unfiltered, and `_in_service!` handles it.
function _matpower_tables(path::String, ::Type{T}) where {T}
    # An element at an isolated bus whose id is the file's lowest or highest falls
    # outside ExaPowerIO's id-to-position table, and its lookup raises a
    # BoundsError: that relies on the lookup being bounds-checked, which the test
    # with such a bus ("parse: refused") pins down.
    raw = try
        ExaPowerIO.parse_matpower(T, path)
    catch err
        err isa BoundsError || rethrow()
        nothing
    end
    if raw === nothing || !isempty(raw.storage) ||
       any(g -> g.bus == 0 || (!g.model_poly && g.n > 0), raw.gen) ||
       any(br -> br.f_bus == 0 || br.t_bus == 0, raw.branch)
        return _tables(_exapowerio_rows(path, T), T)
    end
    return _tables(raw.baseMVA, raw.bus, raw.gen, raw.branch, raw.arc, raw.storage, T)
end
parse_ac_power_data(net::PowerIO.BalancedNetwork, ::Type{T}) where {T} =
    _tables(_powerio_rows(net, T), T)
parse_ac_power_data(m::PowerIO.PioModule{PowerIO.BalancedNetwork}, ::Type{T}) where {T} =
    parse_ac_power_data(m.value, T)

# Every row of the case with its status, as ExaPowerIO gives them unfiltered: buses
# referred to by position (storage by bus id), bus VA in degrees as the file has it,
# branch angles in radians, and everything but the statuses per unit.

function _exapowerio_rows(path::String, ::Type{T}) where {T}
    raw = ExaPowerIO.parse_matpower(T, path; filtered = false)
    base = raw.baseMVA
    storage = raw.storage
    for k in eachindex(storage)
        s = storage[k]
        storage[k] = _with(s, (; Pexts = s.Pexts / base, Qexts = s.Qexts / base))
    end
    return (; baseMVA = base, bus = raw.bus, gen = raw.gen, branch = raw.branch, storage)
end

function _powerio_rows(net::PowerIO.BalancedNetwork, ::Type{T}) where {T}
    # `:open` states a zero-impedance branch as an open circuit carrying its
    # charging, as PowerModels and every earlier parser here did.
    raw = PowerIO.to_powerdata(net, T; zero_impedance = :open)
    base = raw.baseMVA
    for g in raw.gen
        g.status != 0 && g.model == 1 && _pwl_error(raw.bus[g.bus].bus_i)
    end
    bus = BusRow{T}[_row(BusRow{T}, merge(b, (; va = rad2deg(b.va)))) for b in raw.bus]
    gen = GenRow{T}[_row(GenRow{T}, g) for g in raw.gen]
    branch = BranchRow{T}[_row(BranchRow{T}, br) for br in raw.branch]
    storage = StorageRow{T}[_row(StorageRow{T}, merge(s, (;
        Pexts = s.Pexts / base, Qexts = s.Qexts / base,
        p_loss = s.p_loss / base, q_loss = s.q_loss / base))) for s in raw.storage]
    return (; baseMVA = base, bus, gen, branch, storage)
end

# MATPOWER's piecewise-linear cost is a list of points, not the quadratic every
# model here prices generation with.
_pwl_error(bus_i) = throw(ArgumentError(
    "generator at bus " * string(bus_i) * " has a piecewise-linear cost; only " *
    "polynomial costs of degree at most 2 are supported"))

# ── The tables ───────────────────────────────────────────────────────────────

_tables(raw, ::Type{T}) where {T} = _tables(raw.baseMVA, _in_service!(raw, T)..., T)

function _tables(baseMVA, bus, gen, branch, arc, storage, ::Type{T}) where {T}
    return (
        baseMVA = T[baseMVA],
        bus = bus,
        gen = gen,
        arc = arc,
        branch = branch,
        storage = storage,
        ref_buses = Int[i for i in eachindex(bus) if bus[i].type == 3],
        vmax = T[b.vmax for b in bus],
        vmin = T[b.vmin for b in bus],
        pmax = T[g.pmax for g in gen],
        pmin = T[g.pmin for g in gen],
        qmax = T[g.qmax for g in gen],
        qmin = T[g.qmin for g in gen],
        angmax = T[br.angmax for br in branch],
        angmin = T[br.angmin for br in branch],
        rate_a = T[a.rate_a for a in arc],
        vm0 = T[b.vm for b in bus],
        va0 = T[deg2rad(b.va) for b in bus],
        pg0 = T[g.pg for g in gen],
        qg0 = T[g.qg for g in gen],
        pdmax = T[s.charge_rating for s in storage],
        pcmax = T[s.discharge_rating for s in storage],
        srating = T[s.thermal_rating for s in storage],
        emax = T[s.energy_rating for s in storage],
    )
end

# A model holds what is in service. Isolated buses and out-of-service generators,
# branches and storage are dropped, and the rest renumbered densely in file order —
# the order the MATPOWER writer counts rows in, so it is part of this function's
# contract. The row vectors are compacted in place: a large case is not copied.
function _in_service!(raw, ::Type{T}) where {T}
    bus, gen, branch, storage = raw.bus, raw.gen, raw.branch, raw.storage
    # Model position of each bus row, 0 for an isolated bus.
    pos = zeros(Int, length(bus))
    # Taken before `bus` is compacted: the ids of isolated buses for the error
    # message, and the position of every bus id for the storage lookup.
    isolated_id = Dict{Int,Int}(k => bus[k].bus_i for k in eachindex(bus) if bus[k].type == 4)
    _at(k, what) = pos[k] == 0 ?
        throw(ArgumentError("an in-service " * what * " is connected to isolated bus " *
                            string(isolated_id[k]))) : pos[k]
    by_id = isempty(storage) ? Dict{Int,Int}() : Dict{Int,Int}(bus[k].bus_i => k for k in eachindex(bus))

    n = 0
    for k in eachindex(bus)
        bus[k].type == 4 && continue
        n += 1
        pos[k] = n
        # Rows are rewritten only where something changes: most cases have nothing to drop.
        b = bus[k]
        (k != n || b.i != n) && (bus[n] = _with(b, (; i = n)))
    end
    resize!(bus, n)

    n = 0
    for (k, g) in enumerate(gen)
        g.status == 0 && continue
        b = _at(g.bus, "generator")
        !g.model_poly && g.n > 0 && _pwl_error(bus[b].bus_i)
        n += 1
        (k != n || g.i != n || g.bus != b) && (gen[n] = _with(g, (; i = n, bus = b)))
    end
    resize!(gen, n)

    m = count(br -> br.status != 0, branch)
    n = 0
    for (k, br) in enumerate(branch)
        br.status == 0 && continue
        n += 1
        f, t = _at(br.f_bus, "branch"), _at(br.t_bus, "branch")
        rate_a = isfinite(br.rate_a) && br.rate_a > 0 ? br.rate_a :
            _thermal_limit(br.br_r, br.br_x, br.angmin, br.angmax, bus[f].vmax, bus[t].vmax)
        (k != n || br.i != n || br.f_bus != f || br.t_bus != t || br.rate_a != rate_a ||
         br.f_idx != n || br.t_idx != n + m) &&
            (branch[n] = _with(br, (; i = n, f_bus = f, t_bus = t, rate_a, f_idx = n, t_idx = n + m)))
    end
    resize!(branch, m)
    # Arc l is the from end of branch l, arc l + m its to end.
    arc = Vector{ArcRow{T}}(undef, 2m)
    for (l, br) in enumerate(branch)
        arc[l] = ArcRow{T}(l, br.f_bus, br.rate_a)
        arc[l + m] = ArcRow{T}(l + m, br.t_bus, br.rate_a)
    end

    # A storage row names its bus by source id, not by position.
    n = 0
    for s in storage
        s.status == 0 && continue
        k = get(by_id, s.storage_bus, 0)
        k == 0 && throw(ArgumentError("storage at unknown bus " * string(s.storage_bus)))
        n += 1
        storage[n] = _with(s, (; i = n, storage_bus = _at(k, "storage unit")))
    end
    resize!(storage, n)
    return bus, gen, branch, arc, storage
end

# MATPOWER reads a `rateA` of 0 as no limit. Like PowerModels'
# `calc_thermal_limits!`, it becomes the limit the voltage and angle-difference
# bounds already imply. The angle difference is capped at pi, where the voltage
# difference is largest, so a case with MATPOWER's unconstrained -360/360 gets
# the widest such limit rather than one near 0. A source that spells "no
# limit" as Inf gets the same treatment.
function _thermal_limit(r::T, x::T, angmin::T, angmax::T, fr_vmax::T, to_vmax::T) where {T}
    theta_max = min(max(abs(angmin), abs(angmax)), T(pi))
    z2 = r^2 + x^2
    y_mag = iszero(z2) ? zero(T) : inv(sqrt(z2))
    m_vmax = max(fr_vmax, to_vmax)
    c_max = sqrt(fr_vmax^2 + to_vmax^2 - 2 * fr_vmax * to_vmax * cos(theta_max))
    return y_mag * m_vmax * c_max
end
