# A GOC3 problem, read through PowerIO.
#
# `PowerIO.parse` reads a GOC3 problem file as an `AcScucInstance`. Its `network`
# holds the buses, the AC lines and two-winding transformers, the producing and
# consuming devices (as generators and loads), the shunts and the DC lines; its
# `inputs` hold the scheduling data beside them: device time series, reserve
# zones, contingencies and violation costs. Every table keeps the document's
# order and names each element by its GOC3 uid, so every index this model uses
# is a position found by enumerating a table, never a number read out of a uid.

# The uid of a network element.
_goc3_uid(e) = e.component_id::String

"""
    goc3_static(filename) -> NamedTuple

The time-invariant tables of the GOC3 problem in `filename`, indexed the way the
model reads them:

- `bus`: `i`, the bus position, and its voltage bounds.
- `acl_branch` and `acx_branch`: AC lines (`j_ln`) and transformers (`j_xf`) with
  their end buses, series admittance, charging, terminal shunts and ratings;
  `vpd`, `fpd`, `vwr`, `fwr` split the transformers by variable or fixed phase
  shift and winding ratio.
- `dc_branch` (`j_dc`) and `shunt` (`j_sh`).
- `prod` and `cons`: one row per device, `j_prcs` its position among all
  devices and `j_pr` (`j_cs`) among the producers (consumers), with its bus and
  initial active power; `devices` holds PowerIO's device records, whose time
  series the model reads by `j_prcs`.
- The reserve zones and, per zone, the devices at its buses; `outaged`, the
  uids each contingency takes out of service; `dt`, the interval durations; and
  the violation costs.
- `uids`: the uid at each position of each table, for the solution writer.
"""
function goc3_static(filename)
    m = PowerIO.parse(filename)
    m.value isa PowerIO.AcScucInstance ||
        throw(ArgumentError(filename * " is not a GOC3 problem file"))
    inst = m.value::PowerIO.AcScucInstance
    net = inst.network::PowerIO.BalancedNetwork
    inputs = inst.inputs::PowerIO.ScucInputs
    base = net.base_mva::Float64

    # Network elements name their buses by id, reserve zones by uid.
    buses = collect(net.buses)
    bus_of_id = Dict(b.id => i for (i, b) in enumerate(buses))
    bus_of_uid = Dict(_goc3_uid(b) => i for (i, b) in enumerate(buses))
    bus = [(i = i, v_min = b.vmin_pu, v_max = b.vmax_pu) for (i, b) in enumerate(buses)]

    # Lines come first in the branch table, then transformers; a transformer is
    # the branch with a tap and phase-shift control row.
    control = Dict(c.id.local_id => c for c in inputs.transformer_controls)
    branches = collect(net.branches)
    lines = [br for br in branches if !haskey(control, _goc3_uid(br))]
    xfs = [br for br in branches if haskey(control, _goc3_uid(br))]
    # PowerIO folds half the line charging into each terminal susceptance, beside
    # the GOC3 `additional_shunt` terms; the model keeps the two apart. Ratings
    # are in MVA, the model's in per unit.
    function ac_branch(br)
        r, x, b_ch = br.resistance_pu, br.reactance_pu, br.total_charging_susceptance_pu
        return (fr_bus = bus_of_id[br.from_bus_id], to_bus = bus_of_id[br.to_bus_id],
                s_max = br.rate_a_mva / base, s_max_ctg = br.rate_c_mva / base,
                g_sr = r / (x^2 + r^2), b_sr = -x / (x^2 + r^2), b_ch = b_ch,
                g_fr = br.from_conductance_pu, g_to = br.to_conductance_pu,
                b_fr = br.from_susceptance_pu - b_ch / 2, b_to = br.to_susceptance_pu - b_ch / 2)
    end
    acl_branch = [(; j_ln, ac_branch(br)...) for (j_ln, br) in enumerate(lines)]
    acx_branch = [(; j_xf, ac_branch(br)...) for (j_xf, br) in enumerate(xfs)]

    ctl = [control[_goc3_uid(br)] for br in xfs]
    vpd = [(j_xf = j, phi_min = c.phase_shift_min_radians, phi_max = c.phase_shift_max_radians)
           for (j, c) in enumerate(ctl) if c.phase_shift_min_radians < c.phase_shift_max_radians]
    fpd = [(j_xf = j, phi_o = deg2rad(xfs[j].phase_shift_degrees))
           for (j, c) in enumerate(ctl) if !(c.phase_shift_min_radians < c.phase_shift_max_radians)]
    vwr = [(j_xf = j, tau_min = c.tap_ratio_min, tau_max = c.tap_ratio_max)
           for (j, c) in enumerate(ctl) if c.tap_ratio_min < c.tap_ratio_max]
    fwr = [(j_xf = j, tau_o = xfs[j].tap_ratio)
           for (j, c) in enumerate(ctl) if !(c.tap_ratio_min < c.tap_ratio_max)]

    hvdc = collect(net.hvdc)
    dc_branch = [(j_dc = j, fr_bus = bus_of_id[d.from_bus_id], to_bus = bus_of_id[d.to_bus_id],
                  pdc_max = d.maximum_active_power_mw / base,
                  qdc_fr_min = d.minimum_from_reactive_power_mvar / base,
                  qdc_fr_max = d.maximum_from_reactive_power_mvar / base,
                  qdc_to_min = d.minimum_to_reactive_power_mvar / base,
                  qdc_to_max = d.maximum_to_reactive_power_mvar / base) for (j, d) in enumerate(hvdc)]

    shunt_bus = Dict(_goc3_uid(s) => bus_of_id[s.bus_id] for s in net.shunts)
    shunt = [(j_sh = j, bus = shunt_bus[s.id.local_id], g_sh = s.conductance_per_step_pu,
              b_sh = s.susceptance_per_step_pu) for (j, s) in enumerate(inputs.shunts)]

    # A producer is a generator of the network, a consumer a load; either names
    # its bus and its initial active power (in MW, which PowerIO also scales by
    # the base: a ULP from the document's per-unit value).
    devices = inputs.devices
    generator = Dict(_goc3_uid(g) => g for g in net.generators)
    load = Dict(_goc3_uid(l) => l for l in net.loads)
    prod = @NamedTuple{j_prcs::Int, j_pr::Int, bus::Int, p_0::Float64}[]
    cons = @NamedTuple{j_prcs::Int, j_cs::Int, bus::Int, p_0::Float64}[]
    for (j_prcs, d) in enumerate(devices)
        uid = d.id.local_id
        if d.kind == "producer"
            g = generator[uid]
            push!(prod, (j_prcs = j_prcs, j_pr = length(prod) + 1, bus = bus_of_id[g.bus_id],
                         p_0 = g.active_power_mw / base))
        elseif d.kind == "consumer"
            l = load[uid]
            push!(cons, (j_prcs = j_prcs, j_cs = length(cons) + 1, bus = bus_of_id[l.bus_id],
                         p_0 = l.p_mw / base))
        else
            throw(ArgumentError("device " * uid * " is a " * d.kind * ", not a producer or consumer"))
        end
    end

    # A zone's reserve is offered by the devices at its buses: `pr_at[i]` lists
    # the `j_pr` of the producers at bus `i`, `cs_at[i]` the `j_cs` of the consumers.
    function at_bus(rows)
        v = [Int[] for _ in buses]
        for (k, r) in enumerate(rows)
            push!(v[r.bus], k)
        end
        return v
    end
    pr_at, cs_at = at_bus(prod), at_bus(cons)
    azr, rzr = inputs.active_reserve_zones, inputs.reactive_reserve_zones
    L_N_p = length(azr)
    zone_buses(z) = (bus_of_uid[b.local_id] for b in z.buses)
    active_reserve_set_pr = [(n = n, n_p = n, j_pr = j)
                             for (n, z) in enumerate(azr) for i in zone_buses(z) for j in pr_at[i]]
    active_reserve_set_cs = [(n = n, n_p = n, j_cs = j)
                             for (n, z) in enumerate(azr) for i in zone_buses(z) for j in cs_at[i]]
    reactive_reserve_set_pr = [(n = n + L_N_p, n_q = n, j_pr = j)
                               for (n, z) in enumerate(rzr) for i in zone_buses(z) for j in pr_at[i]]
    reactive_reserve_set_cs = [(n = n + L_N_p, n_q = n, j_cs = j)
                               for (n, z) in enumerate(rzr) for i in zone_buses(z) for j in cs_at[i]]

    uids = (
        bus = String[_goc3_uid(b) for b in buses],
        ln = String[_goc3_uid(br) for br in lines],
        xf = String[_goc3_uid(br) for br in xfs],
        dc = String[_goc3_uid(d) for d in hvdc],
        sh = String[s.id.local_id for s in inputs.shunts],
        pr = String[devices[p.j_prcs].id.local_id for p in prod],
        cs = String[devices[c.j_prcs].id.local_id for c in cons],
    )

    return (;
        dt = inputs.interval_durations,
        bus, acl_branch, acx_branch, vpd, fpd, vwr, fwr, dc_branch, shunt,
        devices, prod, cons,
        active_reserve_zones = azr, reactive_reserve_zones = rzr,
        active_reserve_set_pr, active_reserve_set_cs, reactive_reserve_set_pr, reactive_reserve_set_cs,
        outaged = [Set(c.local_id for c in ctg.components) for ctg in inputs.contingencies],
        violation_costs = inputs.violation_costs,
        uids,
    )
end
