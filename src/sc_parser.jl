# The arrays the GOC3 model iterates over, built from the problem's static tables
# (`goc3_static`) and the unit commitment the model is solved at, and the writer
# that puts a solution back into that commitment's file.
#
# The model keeps the commitment fixed: every device, AC line, transformer and
# shunt takes its status (and step) in each interval from a GOC3 solution file.

"""
    goc3_commitment(uc_filename, static) -> NamedTuple

The unit commitment in the GOC3 solution file `uc_filename`, as matrices by
position and interval: device on, startup and shutdown status (rows by
`j_prcs`), AC line and transformer on status, and shunt steps. Rows are joined
to the problem `static` (from [`goc3_static`](@ref)) by uid. An element the
problem has and the file lacks is refused: skipping it would shift every later
row onto the wrong element.
"""
function goc3_commitment(uc_filename, static)
    uc = JSON.parsefile(uc_filename)["time_series_output"]
    L_T = length(static.dt)
    function table(section, uids, field)
        by_uid = Dict(String(r["uid"]) => r for r in uc[section])
        out = Matrix{Int}(undef, length(uids), L_T)
        for (k, uid) in enumerate(uids)
            r = get(by_uid, uid, nothing)
            r === nothing && throw(ArgumentError(uc_filename * " has no " * section * " " * uid))
            out[k, :] .= r[field]
        end
        return out
    end
    devices = static.devices
    on = table("simple_dispatchable_device", [d.id.local_id for d in devices], "on_status")
    # A startup is an interval on after one off, a shutdown the reverse; the first
    # interval follows the device's initial status.
    su, sd = zeros(Int, size(on)), zeros(Int, size(on))
    for (j, d) in enumerate(devices), t in 1:L_T
        before = t == 1 ? Int(d.initial_on_status) : on[j, t-1]
        su[j, t] = before == 0 && on[j, t] == 1
        sd[j, t] = before == 1 && on[j, t] == 0
    end
    return (; on, su, sd,
            ln_on = table("ac_line", static.uids.ln, "on_status"),
            xf_on = table("two_winding_transformer", static.uids.xf, "on_status"),
            sh_step = table("shunt", static.uids.sh, "step"))
end

# The reactive capability of a device, as the rows of constraints 114-115 and
# 124-125 (bounded) or 116 and 126 (linear) read it.
_q_bounded(q) = (beta_max = q.slope_max::Float64, beta_min = q.slope_min::Float64,
                 q_max_p0 = q.reactive_power_at_zero_active_power_max_pu::Float64,
                 q_min_p0 = q.reactive_power_at_zero_active_power_min_pu::Float64)
_q_linear(q) = (beta = q.slope::Float64, q_p0 = q.reactive_power_at_zero_active_power_pu::Float64)

"""
    parse_sc_data(static, uc) -> (sc_time_data, lengths)

The time-indexed arrays of the GOC3 model for the problem `static` (from
[`goc3_static`](@ref)) at the commitment `uc` (from [`goc3_commitment`](@ref)),
and the sizes of its index sets.
"""
function parse_sc_data(s, uc)
    dt = s.dt
    L_T = length(dt)
    periods = 1:L_T
    L_J_ln, L_J_xf, L_J_dc = length(s.acl_branch), length(s.acx_branch), length(s.dc_branch)
    L_J_ac = L_J_ln + L_J_xf
    L_J_br = L_J_ac + L_J_dc
    L_J_pr, L_J_cs = length(s.prod), length(s.cons)
    L_J_cspr = L_J_pr + L_J_cs
    L_J_sh = length(s.shunt)
    I = length(s.bus)
    L_N_p, L_N_q = length(s.active_reserve_zones), length(s.reactive_reserve_zones)
    K = length(s.outaged)
    ε_time = 1e-6

    # Interval end, start and midpoint, in hours. The end is `sum(dt[1:t])`, not a
    # running sum, which rounds differently.
    a_end = [sum(dt[1:t]) for t in periods]
    a_start = a_end .- dt
    a_mid = (a_start .+ a_end) ./ 2

    # Constraints 69 and 112/113 (122-126 for consumers) need two sums per device
    # and interval t over the later intervals t′ whose startup trajectory, ramping
    # up at the startup rate, is still above zero at t: the trajectory power times
    # the startup at t′, and the startups alone.
    function startup_sums(rows)
        p, u = zeros(length(rows), L_T), zeros(length(rows), L_T)
        for (k, r) in enumerate(rows)
            d = s.devices[r.j_prcs]
            for t in periods, tp in (t+1):L_T
                p_supc = d.periods[tp].active_power_min_pu - d.ramp_limits.startup_pu_per_hour * (a_end[tp] - a_end[t])
                p_supc > 0 || continue
                p[k, t] += p_supc * uc.su[r.j_prcs, tp]
                u[k, t] += uc.su[r.j_prcs, tp]
            end
        end
        return p, u
    end
    # Constraint 70, the shutdown half: from the minimum power of the interval
    # before t′ (the initial power for t′ = 1) down at the shutdown rate.
    function shutdown_sums(rows)
        p, u = zeros(length(rows), L_T), zeros(length(rows), L_T)
        for (k, r) in enumerate(rows)
            d = s.devices[r.j_prcs]
            for t in periods, tp in 1:t
                from = tp == 1 ? r.p_0 : d.periods[tp-1].active_power_min_pu
                p_sdpc = from - d.ramp_limits.shutdown_pu_per_hour * (a_end[t] - a_start[tp])
                p_sdpc > 0 || continue
                p[k, t] += p_sdpc * uc.sd[r.j_prcs, tp]
                u[k, t] += uc.sd[r.j_prcs, tp]
            end
        end
        return p, u
    end
    supc_pr, supc2_pr = startup_sums(s.prod)
    sdpc_pr, sdpc2_pr = shutdown_sums(s.prod)
    supc_cs, supc2_cs = startup_sums(s.cons)
    sdpc_cs, sdpc2_cs = shutdown_sums(s.cons)

    # One device in one interval: its ramp and reserve limits, the interval's
    # power bounds and reserve costs, and the commitment.
    function device_t(r, t)
        d = s.devices[r.j_prcs]
        per, ramp, lim = d.periods[t], d.ramp_limits, d.reserve_limits
        c = per.reserve_costs
        return (j_prcs = r.j_prcs, bus = r.bus, p_0 = r.p_0,
            p_ru = ramp.up_pu_per_hour, p_rd = ramp.down_pu_per_hour,
            p_ru_su = ramp.startup_pu_per_hour, p_rd_sd = ramp.shutdown_pu_per_hour,
            c_rgu = c.regulation_up, c_rgd = c.regulation_down, c_scr = c.synchronized, c_nsc = c.nonsynchronized,
            c_rru_on = c.ramping_up_online, c_rru_off = c.ramping_up_offline,
            c_rrd_on = c.ramping_down_online, c_rrd_off = c.ramping_down_offline,
            c_qru = c.reactive_up, c_qrd = c.reactive_down,
            p_rgu_max = lim.regulation_up_pu, p_rgd_max = lim.regulation_down_pu,
            p_scr_max = lim.synchronized_pu, p_nsc_max = lim.nonsynchronized_pu,
            p_rru_on_max = lim.ramping_up_online_pu, p_rru_off_max = lim.ramping_up_offline_pu,
            p_rrd_on_max = lim.ramping_down_online_pu, p_rrd_off_max = lim.ramping_down_offline_pu,
            p_max = per.active_power_max_pu, p_min = per.active_power_min_pu,
            q_max = per.reactive_power_max_pu, q_min = per.reactive_power_min_pu,
            u_on = uc.on[r.j_prcs, t], u_su = uc.su[r.j_prcs, t], u_sd = uc.sd[r.j_prcs, t],
            t = t, dt = dt[t])
    end
    capability(rows, kind) = [r for r in rows if s.devices[r.j_prcs].reactive_capability.kind == kind]
    q_cap(r) = s.devices[r.j_prcs].reactive_capability

    # The energy windows of one device class: a row per window, numbered across
    # the class, and one per interval whose midpoint falls inside it.
    function energy_windows(rows, bounds)
        W = @NamedTuple{w::Int, j::Int, e::Float64}[]
        T = @NamedTuple{w::Int, j::Int, t::Int, dt::Float64}[]
        for (j, r) in enumerate(rows), b in bounds(s.devices[r.j_prcs])
            push!(W, (w = length(W) + 1, j = j, e = b.energy_pu))
            for t in periods
                if b.start_time_hours + ε_time < a_mid[t] && a_mid[t] <= b.end_time_hours + ε_time
                    push!(T, (w = length(W), j = j, t = t, dt = dt[t]))
                end
            end
        end
        return W, T
    end
    W_max_pr, T_max_pr = energy_windows(s.prod, d -> d.energy_upper_bounds)
    W_max_cs, T_max_cs = energy_windows(s.cons, d -> d.energy_upper_bounds)
    W_min_pr, T_min_pr = energy_windows(s.prod, d -> d.energy_lower_bounds)
    W_min_cs, T_min_cs = energy_windows(s.cons, d -> d.energy_lower_bounds)

    # The blocks of each device's energy cost curve, flattened over (device, t, m).
    function cost_blocks(rows)
        out = @NamedTuple{flat_k::Int, j::Int, t::Int, m::Int, c_en::Float64, p_max::Float64}[]
        for (j, r) in enumerate(rows), t in periods
            for (m, b) in enumerate(s.devices[r.j_prcs].periods[t].energy_cost_blocks)
                push!(out, (flat_k = length(out) + 1, j = j, t = t, m = m, c_en = b.marginal_cost, p_max = b.block_size_pu))
            end
        end
        return out
    end
    p_jtm_flattened_pr = [(flat_k = b.flat_k, j_pr = b.j, t = b.t, m = b.m, c_en = b.c_en, p_max = b.p_max)
                          for b in cost_blocks(s.prod)]
    p_jtm_flattened_cs = [(flat_k = b.flat_k, j_cs = b.j, t = b.t, m = b.m, c_en = b.c_en, p_max = b.p_max)
                          for b in cost_blocks(s.cons)]

    # Post-contingency rows: each branch a contingency leaves in service, in every
    # interval, numbered across contingencies.
    survivors(rows, uids, j) = [(ctg = ctg, t = t, b = b) for (ctg, out) in enumerate(s.outaged)
                                for t in periods for b in rows if !(uids[j(b)] in out)]
    jtk_ln_flattened = [(flat_jtk_ln = f, ctg = x.ctg, j_ln = x.b.j_ln, to_bus = x.b.to_bus, fr_bus = x.b.fr_bus,
                         b_sr = x.b.b_sr, s_max_ctg = x.b.s_max_ctg, u_on = uc.ln_on[x.b.j_ln, x.t], t = x.t, dt = dt[x.t])
                        for (f, x) in enumerate(survivors(s.acl_branch, s.uids.ln, b -> b.j_ln))]
    jtk_xf_flattened = [(flat_jtk_xf = f, ctg = x.ctg, j_xf = x.b.j_xf, to_bus = x.b.to_bus, fr_bus = x.b.fr_bus,
                         b_sr = x.b.b_sr, s_max_ctg = x.b.s_max_ctg, u_on = uc.xf_on[x.b.j_xf, x.t], t = x.t, dt = dt[x.t])
                        for (f, x) in enumerate(survivors(s.acx_branch, s.uids.xf, b -> b.j_xf))]
    jtk_dc_flattened = [(flat_jtk_dc = f, ctg = x.ctg, j_dc = x.b.j_dc, to_bus = x.b.to_bus, fr_bus = x.b.fr_bus,
                         t = x.t, dt = dt[x.t])
                        for (f, x) in enumerate(survivors(s.dc_branch, s.uids.dc, b -> b.j_dc))]

    vio = s.violation_costs
    sc_time_data = (
        ;
        periods = periods,

        c_p = [vio.active_power_balance],
        c_q = [vio.reactive_power_balance],
        c_s = [vio.branch_thermal_limit],
        c_e = [vio.energy_requirement],
        busarray = [(; b..., t = t, dt = dt[t]) for b in s.bus, t in periods],
        k_busarray = [(; b..., t = t, k = k) for b in s.bus, t in periods, k in 1:K],
        shuntarray = vec([(; sh..., t = t, u_sh = uc.sh_step[sh.j_sh, t]) for sh in s.shunt, t in periods]),
        k_shuntarray = [(; sh..., t = t, k = k) for sh in s.shunt, t in periods, k in 1:K],

        preservearray = [(n = n, n_p = n,
                          c_rgu = z.regulation_up_violation_cost, c_rgd = z.regulation_down_violation_cost,
                          c_scr = z.synchronized_violation_cost, c_nsc = z.nonsynchronized_violation_cost,
                          c_rru = z.ramping_up_violation_cost, c_rrd = z.ramping_down_violation_cost,
                          σ_rgu = z.regulation_up_requirement_fraction, σ_rgd = z.regulation_down_requirement_fraction,
                          σ_scr = z.synchronized_requirement_fraction, σ_nsc = z.nonsynchronized_requirement_fraction,
                          p_rru_min = z.ramping_up_requirement_pu[t], p_rrd_min = z.ramping_down_requirement_pu[t],
                          t = t, dt = dt[t]) for (n, z) in enumerate(s.active_reserve_zones), t in periods],
        qreservearray = [(n = n + L_N_p, n_q = n,
                          c_qru = z.reactive_up_violation_cost, c_qrd = z.reactive_down_violation_cost,
                          q_qru_min = z.reactive_up_requirement_pu[t], q_qrd_min = z.reactive_down_requirement_pu[t],
                          t = t, dt = dt[t]) for (n, z) in enumerate(s.reactive_reserve_zones), t in periods],

        # Device-fastest, so the first L_J_pr (L_J_cs) rows are the first interval.
        k_prarray = [(j_prcs = r.j_prcs, j_pr = r.j_pr, bus = r.bus, t = t, k = k) for r in s.prod, t in periods, k in 1:K],
        prarray = vec([(; j_pr = r.j_pr, device_t(r, t)...,
                        sum_T_supc_pr_jt = supc_pr[r.j_pr, t], sum_T_sdpc_pr_jt = sdpc_pr[r.j_pr, t],
                        sum2_T_supc_pr_jt = supc2_pr[r.j_pr, t], sum2_T_sdpc_pr_jt = sdpc2_pr[r.j_pr, t])
                       for r in s.prod, t in periods]),
        prarray_pqbounds = vec([(; j_pr = r.j_pr, u_on = uc.on[r.j_prcs, t],
                                 sum2_T_supc_pr_jt = supc2_pr[r.j_pr, t], sum2_T_sdpc_pr_jt = sdpc2_pr[r.j_pr, t],
                                 _q_bounded(q_cap(r))..., t = t) for r in capability(s.prod, "bounded"), t in periods]),
        prarray_pqe = vec([(; j_pr = r.j_pr, u_on = uc.on[r.j_prcs, t],
                            sum2_T_supc_pr_jt = supc2_pr[r.j_pr, t], sum2_T_sdpc_pr_jt = sdpc2_pr[r.j_pr, t],
                            _q_linear(q_cap(r))..., t = t) for r in capability(s.prod, "linear"), t in periods]),

        k_csarray = [(j_prcs = r.j_prcs, j_cs = r.j_cs, bus = r.bus, t = t, k = k) for r in s.cons, t in periods, k in 1:K],
        csarray = vec([(; j_cs = r.j_cs, device_t(r, t)...,
                        sum_T_supc_cs_jt = supc_cs[r.j_cs, t], sum_T_sdpc_cs_jt = sdpc_cs[r.j_cs, t],
                        sum2_T_supc_cs_jt = supc2_cs[r.j_cs, t], sum2_T_sdpc_cs_jt = sdpc2_cs[r.j_cs, t])
                       for r in s.cons, t in periods]),
        csarray_pqbounds = vec([(; j_cs = r.j_cs, u_on = uc.on[r.j_prcs, t],
                                 sum2_T_supc_cs_jt = supc2_cs[r.j_cs, t], sum2_T_sdpc_cs_jt = sdpc2_cs[r.j_cs, t],
                                 _q_bounded(q_cap(r))..., t = t) for r in capability(s.cons, "bounded"), t in periods]),
        csarray_pqe = vec([(; j_cs = r.j_cs, u_on = uc.on[r.j_prcs, t],
                            sum2_T_supc_cs_jt = supc2_cs[r.j_cs, t], sum2_T_sdpc_cs_jt = sdpc2_cs[r.j_cs, t],
                            _q_linear(q_cap(r))..., t = t) for r in capability(s.cons, "linear"), t in periods]),

        acxbrancharray = vec([(; b..., u_on = uc.xf_on[b.j_xf, t], t = t, dt = dt[t]) for b in s.acx_branch, t in periods]),
        aclbrancharray = vec([(; b..., u_on = uc.ln_on[b.j_ln, t], t = t, dt = dt[t]) for b in s.acl_branch, t in periods]),

        fpdarray = [(; b..., t = t) for b in s.fpd, t in periods],
        fwrarray = [(; b..., t = t) for b in s.fwr, t in periods],
        vpdarray = [(; b..., t = t) for b in s.vpd, t in periods],
        vwrarray = [(; b..., t = t) for b in s.vwr, t in periods],
        dclinearray = [(; b..., t = t) for b in s.dc_branch, t in periods],

        p_jt_fr_dc_max = [dc.pdc_max for dc in s.dc_branch, t in periods],
        p_jt_to_dc_max = [dc.pdc_max for dc in s.dc_branch, t in periods],
        q_jt_fr_dc_lvar = [dc.qdc_fr_min for dc in s.dc_branch, t in periods],
        q_jt_fr_dc_uvar = [dc.qdc_fr_max for dc in s.dc_branch, t in periods],
        q_jt_to_dc_lvar = [dc.qdc_to_min for dc in s.dc_branch, t in periods],
        q_jt_to_dc_uvar = [dc.qdc_to_max for dc in s.dc_branch, t in periods],

        preservesetarray_pr = [(; b..., t = t) for b in s.active_reserve_set_pr, t in periods],
        preservesetarray_cs = [(; b..., t = t) for b in s.active_reserve_set_cs, t in periods],
        qreservesetarray_pr = [(; b..., t = t) for b in s.reactive_reserve_set_pr, t in periods],
        qreservesetarray_cs = [(; b..., t = t) for b in s.reactive_reserve_set_cs, t in periods],

        v_lvar = repeat([b.v_min for b in s.bus], 1, L_T),
        v_uvar = repeat([b.v_max for b in s.bus], 1, L_T),

        W_en_max_pr = [(w_en_max_pr_ind = w.w, j_pr = w.j, e_max = w.e) for w in W_max_pr],
        W_en_max_cs = [(w_en_max_cs_ind = w.w, j_cs = w.j, e_max = w.e) for w in W_max_cs],
        T_w_en_max_pr = [(w_en_max_pr_ind = w.w, j_pr = w.j, t = w.t, dt = w.dt) for w in T_max_pr],
        T_w_en_max_cs = [(w_en_max_cs_ind = w.w, j_cs = w.j, t = w.t, dt = w.dt) for w in T_max_cs],
        W_en_min_pr = [(w_en_min_pr_ind = w.w, j_pr = w.j, e_min = w.e) for w in W_min_pr],
        W_en_min_cs = [(w_en_min_cs_ind = w.w, j_cs = w.j, e_min = w.e) for w in W_min_cs],
        T_w_en_min_pr = [(w_en_min_pr_ind = w.w, j_pr = w.j, t = w.t, dt = w.dt) for w in T_min_pr],
        T_w_en_min_cs = [(w_en_min_cs_ind = w.w, j_cs = w.j, t = w.t, dt = w.dt) for w in T_min_cs],

        tk_index = [(t = t, k = k) for t in periods, k in 1:K],

        p_jtm_flattened_pr = p_jtm_flattened_pr,
        p_jtm_flattened_cs = p_jtm_flattened_cs,
        p_jtm_pr_uvar = [b.p_max for b in p_jtm_flattened_pr],
        p_jtm_cs_uvar = [b.p_max for b in p_jtm_flattened_cs],
        jtk_ln_flattened = jtk_ln_flattened,
        jtk_xf_flattened = jtk_xf_flattened,
        jtk_dc_flattened = jtk_dc_flattened,
    )

    lengths = (; L_J_xf, L_J_ln, L_J_ac, L_J_dc, L_J_br, L_J_cs, L_J_pr, L_J_cspr, L_J_sh, I, L_T, L_N_p, L_N_q,
               L_W_en_min_pr = length(W_min_pr), L_W_en_min_cs = length(W_min_cs),
               L_W_en_max_pr = length(W_max_pr), L_W_en_max_cs = length(W_max_cs), K)

    return sc_time_data, lengths
end

"""
    save_go3_solution(uc_filename, solution_name, result, vars, lengths, uids)

Write the solution `result` of a model `goc3_model(filename, uc_filename)` into
the GOC3 solution file `uc_filename`, and save it as `solution_name`. `vars` and
`lengths` are what `goc3_model` returned, and `uids` the `uids` field of the
`sc_data_array` it returned, which names the element at every position.

Device dispatch and reserves, transformer taps and phase shifts, DC line flows,
and bus voltages are written; AC line flows follow from the voltages, and the
commitment is kept as it was.
"""
function save_go3_solution(uc_filename, solution_name, result, vars, lengths, uids)
    uc_data = JSON.parsefile(uc_filename)
    L_T = lengths.L_T
    sol(v, j) = Array(solution(result, v))[j, :]
    position(uids) = Dict(uid => j for (j, uid) in enumerate(uids))
    pr, cs, xf, dc, bus = position(uids.pr), position(uids.cs), position(uids.xf), position(uids.dc), position(uids.bus)
    function at(table, line)
        j = get(table, line["uid"], 0)
        j == 0 && throw(ArgumentError(uc_filename * " names " * line["uid"] * ", which the problem does not have"))
        return j
    end

    for line in uc_data["time_series_output"]["simple_dispatchable_device"]
        if haskey(cs, line["uid"])
            j = cs[line["uid"]]
            line["p_syn_res"] = sol(vars.p_jt_scr_cs, j)
            line["p_ramp_res_up_online"] = sol(vars.p_jt_rru_on_cs, j)
            line["p_nsyn_res"] = zeros(L_T)
            line["p_reg_res_up"] = sol(vars.p_jt_rgu_cs, j)
            line["p_ramp_res_down_online"] = sol(vars.p_jt_rrd_on_cs, j)
            line["p_on"] = sol(vars.p_jt_on_cs, j)
            line["q"] = sol(vars.q_jt_cs, j)
            line["p_reg_res_down"] = sol(vars.p_jt_rgd_cs, j)
            line["p_ramp_res_up_offline"] = zeros(L_T)
            line["q_res_down"] = sol(vars.q_jt_qrd_cs, j)
            line["q_res_up"] = sol(vars.q_jt_qru_cs, j)
            line["p_ramp_res_down_offline"] = sol(vars.p_jt_rrd_off_cs, j)
        else
            j = at(pr, line)
            line["p_syn_res"] = sol(vars.p_jt_scr_pr, j)
            line["p_ramp_res_up_online"] = sol(vars.p_jt_rru_on_pr, j)
            line["p_nsyn_res"] = sol(vars.p_jt_nsc_pr, j)
            line["p_reg_res_up"] = sol(vars.p_jt_rgu_pr, j)
            line["p_ramp_res_down_online"] = sol(vars.p_jt_rrd_on_pr, j)
            line["p_on"] = sol(vars.p_jt_on_pr, j)
            line["q"] = sol(vars.q_jt_pr, j)
            line["p_reg_res_down"] = sol(vars.p_jt_rgd_pr, j)
            line["p_ramp_res_up_offline"] = sol(vars.p_jt_rru_off_pr, j)
            line["q_res_down"] = sol(vars.q_jt_qrd_pr, j)
            line["q_res_up"] = sol(vars.q_jt_qru_pr, j)
            line["p_ramp_res_down_offline"] = zeros(L_T)
        end
    end
    for line in uc_data["time_series_output"]["two_winding_transformer"]
        j = at(xf, line)
        line["tm"] = sol(vars.τ_jt_xf, j)
        line["ta"] = sol(vars.φ_jt_xf, j)
    end
    for line in uc_data["time_series_output"]["dc_line"]
        j = at(dc, line)
        line["qdc_fr"] = sol(vars.q_jt_fr_dc, j)
        line["pdc_fr"] = sol(vars.p_jt_fr_dc, j)  # pdc_to follows from conservation
        line["qdc_to"] = sol(vars.q_jt_to_dc, j)
    end
    for line in uc_data["time_series_output"]["bus"]
        j = at(bus, line)
        line["vm"] = sol(vars.v_it, j)
        line["va"] = sol(vars.θ_it, j)
    end

    open(solution_name, "w") do io
        JSON.print(io, uc_data, 4)
    end
end
