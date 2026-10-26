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
# One isbits NamedTuple type per table, with the field names and order the
# model bodies were written against. Naming the row TYPE rather than letting a
# comprehension find it keeps an empty table (case14 has no storage) a
# `Vector{StorageRow{T}}` rather than a `Vector{Union{}}`, which is what a GPU
# backend's `convert_array` and `juliac --trim=safe` both need.
const BusRow{T} = @NamedTuple{
    i::Int, bus_i::Int, type::Int, pd::T, qd::T, gs::T, bs::T, area::Int,
    vm::T, va::T, baseKV::T, zone::Int, vmax::T, vmin::T}
const GenRow{T} = @NamedTuple{
    i::Int, bus::Int, pg::T, qg::T, qmax::T, qmin::T, vg::T, mbase::T, status::Int,
    pmax::T, pmin::T, model_poly::Bool, startup::T, shutdown::T, n::Int, c::NTuple{3,T}}
const BranchRow{T} = @NamedTuple{
    i::Int, f_bus::Int, t_bus::Int, br_r::T, br_x::T, b_fr::T, b_to::T, g_fr::T, g_to::T,
    rate_a::T, rate_b::T, rate_c::T, tap::T, shift::T, status::Int, angmin::T, angmax::T,
    f_idx::Int, t_idx::Int, c1::T, c2::T, c3::T, c4::T, c5::T, c6::T, c7::T, c8::T}
const ArcRow{T} = @NamedTuple{i::Int, bus::Int, rate_a::T}
const StorageRow{T} = @NamedTuple{
    i::Int, storage_bus::Int, Pexts::T, Qexts::T, energy::T, energy_rating::T,
    charge_rating::T, discharge_rating::T, charge_efficiency::T, discharge_efficiency::T,
    thermal_rating::T, qmin::T, qmax::T, Zr::T, Zim::T, p_loss::T, q_loss::T, status::Int}

# The fields of `R` read off a PowerIO row, after `merge` has replaced the ones
# that change: PowerIO's rows carry these names and a few more.
_row(::Type{R}, nt::NamedTuple) where {R<:NamedTuple} = R(NamedTuple{fieldnames(R)}(nt))

# ── Where a case comes from ──────────────────────────────────────────────────

# A case is a file PowerIO can read (MATPOWER, PSS/E, PowerModels JSON, …), or
# a bare PGLib-OPF name such as "pglib_opf_case118_ieee.m", which is looked up
# in the lazily downloaded PGLib-OPF v23.07 artifact. Anything else is refused
# here rather than sent to the artifact, so a mistyped path never triggers the
# download or reports a path the caller did not write.
function _case_path(case::AbstractString)
    path = String(case)
    ispath(path) && return path
    if !isabspath(path) && startswith(basename(path), "pglib_opf_")
        return joinpath(artifact"PGLib_opf", "pglib-opf-23.07", path)
    end
    throw(ArgumentError("no case file at \"" * path * "\" (a bare name is looked up in " *
                        "PGLib-OPF only when it starts with \"pglib_opf_\")"))
end

# A path, or a network PowerIO already parsed: one parsed with a `format`, or
# edited with `PowerIO.apply_updates!`, builds the same model as its file.
# The typed `parse` is what keeps this inferable: the untyped one picks the
# module's value type at run time, which `juliac --trim=safe` refuses.
_network(net::PowerIO.BalancedNetwork) = net
_network(m::PowerIO.PioModule{PowerIO.BalancedNetwork}) = m.value
_network(case::AbstractString) = PowerIO.parse(PowerIO.BalancedNetwork, _case_path(case)).value

# ── The tables ───────────────────────────────────────────────────────────────

# `T` is a TYPE PARAMETER, not a value of type `Type`. As a plain positional
# argument it arrives typed `Type` — abstract — and nothing downstream can
# specialize on it, so every field read off the result widens. Written
# `::Type{T}` it is a static parameter and the whole chain stays concrete.
parse_ac_power_data(case) = parse_ac_power_data(case, Float64)
function parse_ac_power_data(case, ::Type{T}) where {T}
    # `:open` states a zero-impedance branch as an open circuit carrying its
    # charging, as PowerModels and every earlier parser here did.
    raw = PowerIO.to_powerdata(_network(case), T; zero_impedance = :open)
    bus, gen, branch, arc, storage = _in_service(raw, T)
    return (
        baseMVA = T[raw.baseMVA],
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
        va0 = T[b.va for b in bus],
        pg0 = T[g.pg for g in gen],
        qg0 = T[g.qg for g in gen],
        pdmax = T[s.charge_rating for s in storage],
        pcmax = T[s.discharge_rating for s in storage],
        srating = T[s.thermal_rating for s in storage],
        emax = T[s.energy_rating for s in storage],
    )
end

# PowerIO states every row with its status; a model holds what is in service.
# Isolated buses and out-of-service generators, branches and storage are
# dropped, and the rest renumbered densely in file order — the order the
# MATPOWER writer counts rows in, so it is part of this function's contract.
function _in_service(raw, ::Type{T}) where {T}
    # Model position of each PowerIO bus row, 0 for an isolated bus.
    pos = zeros(Int, length(raw.bus))
    bus = BusRow{T}[]
    for (k, b) in enumerate(raw.bus)
        b.type == 4 && continue
        pos[k] = length(bus) + 1
        push!(bus, _row(BusRow{T}, merge(b, (; i = pos[k]))))
    end
    _at(k, what) = pos[k] == 0 ?
        throw(ArgumentError("an in-service " * what * " is connected to isolated bus " *
                            string(raw.bus[k].bus_i))) : pos[k]

    gen = GenRow{T}[]
    for g in raw.gen
        g.status == 0 && continue
        # MATPOWER's piecewise-linear cost is a list of points, not the
        # quadratic every model here prices generation with.
        g.model == 1 && throw(ArgumentError(
            "generator at bus " * string(raw.bus[g.bus].bus_i) * " has a piecewise-linear " *
            "cost; only polynomial costs of degree at most 2 are supported"))
        push!(gen, _row(GenRow{T}, merge(g, (; i = length(gen) + 1, bus = _at(g.bus, "generator")))))
    end

    active = [br for br in raw.branch if br.status != 0]
    m = length(active)
    branch = BranchRow{T}[]
    for (l, br) in enumerate(active)
        f, t = _at(br.f_bus, "branch"), _at(br.t_bus, "branch")
        rate_a = isfinite(br.rate_a) && br.rate_a > 0 ? br.rate_a :
            _thermal_limit(br.br_r, br.br_x, br.angmin, br.angmax, bus[f].vmax, bus[t].vmax)
        push!(branch, _row(BranchRow{T},
            merge(br, (; i = l, f_bus = f, t_bus = t, rate_a, f_idx = l, t_idx = l + m))))
    end
    # Arc l is the from end of branch l, arc l + m its to end.
    arc = ArcRow{T}[(; i = l, bus = br.f_bus, rate_a = br.rate_a) for (l, br) in enumerate(branch)]
    append!(arc, ArcRow{T}[(; i = l + m, bus = br.t_bus, rate_a = br.rate_a) for (l, br) in enumerate(branch)])

    # A storage row names its bus by source id, not by position.
    by_id = Dict{Int,Int}(b.bus_i => k for (k, b) in enumerate(raw.bus))
    storage = StorageRow{T}[]
    for s in raw.storage
        s.status == 0 && continue
        k = get(by_id, s.storage_bus, 0)
        k == 0 && throw(ArgumentError("storage at unknown bus " * string(s.storage_bus)))
        push!(storage, _row(StorageRow{T}, merge(s, (;
            i = length(storage) + 1, storage_bus = _at(k, "storage unit"),
            p_loss = s.p_loss / raw.baseMVA, q_loss = s.q_loss / raw.baseMVA))))
    end
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
