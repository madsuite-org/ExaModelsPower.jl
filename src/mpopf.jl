# `::Type{T}`, not a `Type`-typed value — the same fault the static parser had:
# as a plain argument it arrives abstract and `parse_ac_power_data` cannot
# specialize, so every field read off the result widens.
parse_mp_power_data(case, N, r) = parse_mp_power_data(case, N, r, Float64)
function parse_mp_power_data(case, N, corrective_action_ratio, ::Type{T}) where {T}

    raw = parse_ac_power_data(case, T)

    nbus = length(raw.bus)

    # No empty-storage ternary: its two branches had DIFFERENT types (a vector
    # of one NamedTuple shape against a matrix of another), which made the whole
    # return non-concrete — the same fault the static parser had. A comprehension
    # over empty storage is a well-typed 0xN matrix on its own.
    data = (
        ;
        raw...,
        refarray = [(i,t) for i in raw.ref_buses, t in 1:N],
        barray = [(;b, t = t) for b in raw.branch, t in 1:N ],
        busarray = [(;b, t = t) for b in raw.bus, t in 1:N ],
        arcarray = [(;a, t = t) for a in raw.arc, t in 1:N ],
        genarray = [(;g, t = t) for g in raw.gen, t in 1:N ],
        storarray = [(;s, t = t) for s in raw.storage, t in 1:N],
        branch_rate_a = [br.rate_a for br in raw.branch],
        Δp = corrective_action_ratio .* (raw.pmax .- raw.pmin)
    )

    return data
end

# A load curve scales each period's demand. Bus shunts are fixed admittances
# (capacitor banks, reactors), not demand, so they keep their base value.
function update_load_data(busarray, curve)
    N = size(busarray, 2)
    length(curve) >= N || throw(DimensionMismatch(
        "a load curve of " * string(length(curve)) * " periods was given for " * string(N)))
    for t in 1:N, x in axes(busarray, 1)
        b = busarray[x, t].b
        busarray[x, t] = (; b = merge(b, (; pd = oftype(b.pd, b.pd * curve[t]),
                                             qd = oftype(b.qd, b.qd * curve[t]))), t)
    end
    return busarray
end

# Pd, Qd as input: MW, one row per bus of the model (isolated buses excluded)
# in file order, one column per period.
function update_load_data(busarray, pd, qd, baseMVA)
    nbus, N = size(busarray)
    size(pd) == size(qd) || throw(DimensionMismatch(
        "Pd is " * string(size(pd)) * " but Qd is " * string(size(qd))))
    size(pd, 1) == nbus || throw(DimensionMismatch(
        "the load matrices have " * string(size(pd, 1)) * " rows but the case has " *
        string(nbus) * " buses"))
    size(pd, 2) >= N || throw(DimensionMismatch(
        "the load matrices have " * string(size(pd, 2)) * " periods but " * string(N) *
        " were asked for"))
    for t in 1:N, x in 1:nbus
        b = busarray[x, t].b
        busarray[x, t] = (; b = merge(b, (; pd = oftype(b.pd, pd[x, t] / baseMVA),
                                             qd = oftype(b.qd, qd[x, t] / baseMVA))), t)
    end
    return busarray
end

#If no storage contraints, the "build_base_mpopf" returns the final version of the mpopf
# ── Multi-period: one body per formulation, as in the static case ───────────
#
# The multi-period spine is its OWN — generation, flows, objective, thermal
# limits and the ramp rate all come before the voltage block, where the static
# body puts voltage first — so this is a separate class rather than the static
# body with a time index. Within it, polar and rect differ only in the voltage
# block and in which method of each expression is called, exactly as they do
# statically, so the same `OPFForm` types dispatch here too.
#
# DC is a third formulation rather than a special case: no reactive half, flows
# per branch instead of per arc, ohms law in place of the four branch flows,
# and no thermal limits. The ramp rate is on `pg` and carries over unchanged.

function add_gen_vars_mp!(core, ::Union{Polar,Rect}, data, N)
    @add_var(core, pg, length(data.gen), N; lvar = data.rep.pmin, uvar = data.rep.pmax)
    @add_var(core, qg, length(data.gen), N; lvar = data.rep.qmin, uvar = data.rep.qmax)
    return core, (; pg, qg)
end

function add_gen_vars_mp!(core, ::DC, data, N)
    @add_var(core, pg, length(data.gen), N; lvar = data.rep.pmin, uvar = data.rep.pmax)
    return core, (; pg)
end

function add_flow_vars_mp!(core, ::Union{Polar,Rect}, data, N)
    @add_var(core, p, length(data.arc), N; lvar = data.rep.nrate_a, uvar = data.rep.rate_a)
    @add_var(core, q, length(data.arc), N; lvar = data.rep.nrate_a, uvar = data.rep.rate_a)
    return core, (; p, q)
end

function add_flow_vars_mp!(core, ::DC, data, N)
    @add_var(core, pf, length(data.branch), N;
        lvar = data.rep.nbranch_rate_a, uvar = data.rep.branch_rate_a)
    return core, (; pf)
end

# Thermal limits need both halves of the flow, so they are AC-only.
function add_thermal_mp!(core, ::Union{Polar,Rect}, data, F)
    @add_con(core, c_from_thermal_limit,
        c_thermal_limit(b, F.p[b.f_idx, t], F.q[b.f_idx, t]) for (b, t) in data.barray;
        lcon = data.rep.ninf_b)
    @add_con(core, c_to_thermal_limit,
        c_thermal_limit(b, F.p[b.t_idx, t], F.q[b.t_idx, t]) for (b, t) in data.barray;
        lcon = data.rep.ninf_b)
    return core, (; c_from_thermal_limit, c_to_thermal_limit)
end
add_thermal_mp!(core, ::DC, data, F) = (core, (;))

"""
    MPOPF_DEFAULT_CURVE

The load curve behind [`mpopf_args_default`](@ref): `[1.0, 0.9, 0.8, 0.95, 1.0]`.

A compiled multi-period library needs a ONE-argument, package-owned function to
call — the generated app resolves it by name from another process, so a closure
over a curve cannot be reached. The curve and horizon are compile-time facts of
such a library anyway, so the default lives here as a constant; a different
curve means defining your own one-argument wrapper over [`mpopf_args`](@ref) in
your own package.
"""
const MPOPF_DEFAULT_CURVE = [1.0, 0.9, 0.8, 0.95, 1.0]

# What a multi-period recipe freezes, and what it does not. `N` is structural:
# the body slices `genarray[:, 2:N]` and expands bounds with `repeat(v, 1, N)`,
# neither of which has a symbolic form. The BUS COUNT is not — it is an extent
# of the data, so it comes from `length(data.bus)` exactly as the static
# formulations do it, and one recipe serves every network size.
"""
    mpopf_args_default(case) -> (data,)

[`mpopf_args`](@ref) at [`MPOPF_DEFAULT_CURVE`](@ref) — the spelling a compiled
library can call. Define your own one-argument wrapper in your package for a
different curve.
"""
mpopf_args_default(case) =
    mpopf_args(case, MPOPF_DEFAULT_CURVE, length(MPOPF_DEFAULT_CURVE), Float64, nothing, 0.1)

"""
    mpopf_args(case, curve; N, T, backend, corrective_action_ratio) -> (data,)

The arguments that close [`mpopf_recipe`](@ref): the parsed case with the load
curve already applied, plus everything the recipe cannot compute from a
placeholder.

`N` and the curve are NOT arguments to a compiled library — they are baked into
the recipe. The body slices `genarray[:, 2:N]` and expands bounds with
`repeat(v, 1, N)`, neither of which has a symbolic form, so a compiled
multi-period library is per-`(N, curve)` and the case file is the one value
that crosses the boundary.

The N-expanded bounds are grouped under `rep` rather than spread across the top
level: there are fifteen of them, and `map` over a NamedTuple stops inferring
elementwise past 31 fields, which would leave the whole argument tuple with
concrete names and no value types.
"""
mpopf_args(case, curve; N = length(curve), T = Float64, backend = nothing,
           corrective_action_ratio = 0.1) =
    mpopf_args(case, curve, N, T, backend, corrective_action_ratio)

# The positional method is the one a compiled library reaches: `::Type{T}` makes
# `T` a static parameter, where the keyword form leaves it a `Type`-typed value
# nothing downstream can specialize on.
function mpopf_args(case, curve, N, ::Type{T}, backend, corrective_action_ratio) where {T}
    d = parse_mp_power_data(case, N, corrective_action_ratio, T)
    update_load_data(d.busarray, curve)
    return (convert_data(_mp_narrow(d, N, T), backend),)
end

function _mp_narrow(d, N, ::Type{T}) where {T}
    rep = (;
        pmin = repeat(d.pmin, 1, N), pmax = repeat(d.pmax, 1, N),
        qmin = repeat(d.qmin, 1, N), qmax = repeat(d.qmax, 1, N),
        rate_a = repeat(d.rate_a, 1, N), nrate_a = repeat(-d.rate_a, 1, N),
        branch_rate_a = repeat(d.branch_rate_a, 1, N),
        nbranch_rate_a = repeat(-d.branch_rate_a, 1, N),
        vmin = repeat(d.vmin, 1, N), vmax = repeat(d.vmax, 1, N),
        vmin2 = repeat(d.vmin, 1, N) .^ 2, vmax2 = repeat(d.vmax, 1, N) .^ 2,
        angmin = repeat(d.angmin, 1, N), angmax = repeat(d.angmax, 1, N),
        srating = repeat(d.srating, 1, N), nsrating = repeat(-d.srating, 1, N),
        emax = repeat(d.emax, 1, N),
        pdmax = repeat(d.pdmax, 1, N), pcmax = repeat(d.pcmax, 1, N),
        dp = repeat(d.Δp, 1, N - 1), ndp = repeat(-d.Δp, 1, N - 1),
        ninf_b = fill(T(-Inf), size(d.barray)),
        zero_stor = zeros(T, size(d.storarray)),
        inf_stor = fill(T(Inf), size(d.storarray)),
        ninf_stor = fill(T(-Inf), size(d.storarray)),
        npcmax = repeat(-d.pcmax, 1, N),
    )
    return (;
        bus = d.bus, gen = d.gen, arc = d.arc, branch = d.branch, storage = d.storage,
        refarray = d.refarray, barray = d.barray, busarray = d.busarray,
        arcarray = d.arcarray, genarray = d.genarray, storarray = d.storarray,
        ramparray = d.genarray[:, 2:N],
        storarray_tail = d.storarray[:, 2:N],
        storarray_head = d.storarray[:, 1],
        rep = rep,
    )
end

function build_base_mpopf(core, form, data, N)
    core, G = add_gen_vars_mp!(core, form, data, N)
    core, F = add_flow_vars_mp!(core, form, data, N)

    @add_obj(core, o, gen_cost(g, G.pg[g.i, t]) for (g, t) in data.genarray)

    core, thermal = add_thermal_mp!(core, form, data, F)

    @add_con(core, c_ramp_rate,
        c_ramp(G.pg[g.i, t-1], G.pg[g.i, t]) for (g, t) in data.ramparray;
        lcon = data.rep.ndp,
        ucon = data.rep.dp)

    return core, merge(G, F), merge(thermal, (; c_ramp_rate))
end

# ── the per-formulation halves ──────────────────────────────────────────────

function add_voltage_mp!(core, ::Polar, data, Nbus, N, ::Type{T}) where {T}
    @add_var(core, va, Nbus, N; lvar = -pi, uvar = pi)
    @add_var(core, vm, Nbus, N;
        start = one(T),
        lvar = data.rep.vmin,
        uvar = data.rep.vmax)
    return core, (; va, vm)
end

function add_voltage_mp!(core, ::Rect, data, Nbus, N, ::Type{T}) where {T}
    @add_var(core, vr, Nbus, N; start = one(T))
    @add_var(core, vim, Nbus, N;)
    return core, (; vr, vim)
end

function add_voltage_mp!(core, ::DC, data, Nbus, N, ::Type{T}) where {T}
    @add_var(core, va, Nbus, N; lvar = -pi, uvar = pi)
    return core, (; va)
end

# The same expressions as the static body, read at `[i, t]` rather than `[i]`.
@inline c_ref_mp(::Union{Polar,DC}, V, i, t) = c_ref_angle_polar(V.va[i, t])
@inline c_ref_mp(::Rect, V, i, t) = c_ref_angle_rect(V.vr[i, t], V.vim[i, t])

@inline c_angle_mp(::Union{Polar,DC}, b, V, t) =
    c_phase_angle_diff_polar(b, V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline c_angle_mp(::Rect, b, V, t) =
    c_phase_angle_diff_rect(b, V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])

@inline c_bal_p_mp(::Polar, b, V, t) = c_active_power_balance_demand_polar(b, V.vm[b.i, t])
@inline c_bal_p_mp(::Rect, b, V, t) = c_active_power_balance_demand_rect(b, V.vr[b.i, t], V.vim[b.i, t])
@inline c_bal_p_mp(::DC, b, V, t) = c_active_power_balance_dc(b)
@inline c_bal_q_mp(::Polar, b, V, t) = c_reactive_power_balance_demand_polar(b, V.vm[b.i, t])
@inline c_bal_q_mp(::Rect, b, V, t) = c_reactive_power_balance_demand_rect(b, V.vr[b.i, t], V.vim[b.i, t])

@inline ac_flow_mp(f, s::Symbol, b, F, V, t) = ac_flow_mp(f, Val(s), b, F, V, t)
@inline ac_flow_mp(::Polar, ::Val{:ta}, b, F, V, t) = c_to_active_power_flow_polar(b, F.p[b.f_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:tr}, b, F, V, t) = c_to_reactive_power_flow_polar(b, F.q[b.f_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:fa}, b, F, V, t) = c_from_active_power_flow_polar(b, F.p[b.t_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:fr}, b, F, V, t) = c_from_reactive_power_flow_polar(b, F.q[b.t_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:ta}, b, F, V, t) = c_to_active_power_flow_rect(b, F.p[b.f_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:tr}, b, F, V, t) = c_to_reactive_power_flow_rect(b, F.q[b.f_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:fa}, b, F, V, t) = c_from_active_power_flow_rect(b, F.p[b.t_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:fr}, b, F, V, t) = c_from_reactive_power_flow_rect(b, F.q[b.t_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])

function add_flow_cons_mp!(core, form::Union{Polar,Rect}, data, V, F)
    @add_con(core, c_to_active_power_flow, ac_flow_mp(form, :ta, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_to_reactive_power_flow, ac_flow_mp(form, :tr, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_from_active_power_flow, ac_flow_mp(form, :fa, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_from_reactive_power_flow, ac_flow_mp(form, :fr, b, F, V, t) for (b, t) in data.barray)
    return core, (; c_to_active_power_flow, c_to_reactive_power_flow,
                    c_from_active_power_flow, c_from_reactive_power_flow)
end

function add_flow_cons_mp!(core, ::DC, data, V, F)
    @add_con(core, c_ohms_law,
        c_ohms_law_dcopf(br, F.pf[br.i, t], V.va[br.f_bus, t], V.va[br.t_bus, t])
        for (br, t) in data.barray)
    return core, (; c_ohms_law)
end

# The balance ROWS and the appends into them are separate steps because the
# rectangular form inserts its voltage-magnitude rows BETWEEN them. Appending a
# term adds Jacobian entries, so doing it before those rows exist puts the COO
# triplets in a different order — same model, different ordering, which the
# equivalence check sees. (The static body has the opposite order and is split
# the same way for the same reason.)
function add_balance_cons_mp!(core, form::Union{Polar,Rect}, data, V)
    @add_con(core, c_active_power_balance, c_bal_p_mp(form, b, V, t) for (b, t) in data.busarray)
    @add_con(core, c_reactive_power_balance, c_bal_q_mp(form, b, V, t) for (b, t) in data.busarray)
    return core, (; c_active_power_balance, c_reactive_power_balance)
end

function add_balance_cons_mp!(core, form::DC, data, V)
    @add_con(core, c_active_power_balance, c_bal_p_mp(form, b, V, t) for (b, t) in data.busarray)
    return core, (; c_active_power_balance)
end

function add_balance_appends_mp!(core, ::Union{Polar,Rect}, data, Nbus, B, G, F)
    c_active_power_balance, c_reactive_power_balance = B.c_active_power_balance, B.c_reactive_power_balance
    @add_con!(core, c_active_power_balance, a.bus + Nbus*(t-1) => F.p[a.i, t] for (a, t) in data.arcarray)
    @add_con!(core, c_reactive_power_balance, a.bus + Nbus*(t-1) => F.q[a.i, t] for (a, t) in data.arcarray)
    @add_con!(core, c_active_power_balance, g.bus + Nbus*(t-1) => -G.pg[g.i, t] for (g, t) in data.genarray)
    @add_con!(core, c_reactive_power_balance, g.bus + Nbus*(t-1) => -G.qg[g.i, t] for (g, t) in data.genarray)
    return core
end

function add_balance_appends_mp!(core, ::DC, data, Nbus, B, G, F)
    c_active_power_balance = B.c_active_power_balance
    @add_con!(core, c_active_power_balance, g.bus + Nbus*(t-1) => -G.pg[g.i, t] for (g, t) in data.genarray)
    @add_con!(core, c_active_power_balance, br.f_bus + Nbus*(t-1) => F.pf[br.i, t] for (br, t) in data.barray)
    @add_con!(core, c_active_power_balance, br.t_bus + Nbus*(t-1) => -F.pf[br.i, t] for (br, t) in data.barray)
    return core
end

function add_extras_mp!(core, ::Rect, data, N, V)
    @add_con(core, c_voltage_magnitude,
        c_voltage_magnitude_rect(V.vr[b.i, t], V.vim[b.i, t]) for (b, t) in data.busarray;
        lcon = data.rep.vmin2,
        ucon = data.rep.vmax2)
    return core, (; c_voltage_magnitude)
end
add_extras_mp!(core, ::Union{Polar,DC}, data, N, V) = (core, (;))

function add_mpopf_cons(core, form, data, N, Nbus, vars, cons, ::Type{T} = Float64) where {T}
    core, V = add_voltage_mp!(core, form, data, Nbus, N, T)
    @add_con(core, c_ref_angle, c_ref_mp(form, V, i, t) for (i, t) in data.refarray)
    core, flowcons = add_flow_cons_mp!(core, form, data, V, vars)
    @add_con(core, c_phase_angle_diff, c_angle_mp(form, b, V, t) for (b, t) in data.barray;
        lcon = data.rep.angmin,
        ucon = data.rep.angmax)
    core, balcons = add_balance_cons_mp!(core, form, data, V)
    core, extras = add_extras_mp!(core, form, data, N, V)
    core = add_balance_appends_mp!(core, form, data, Nbus, balcons, vars, vars)
    return core, merge(vars, V),
           merge(cons, (; c_ref_angle), flowcons, (; c_phase_angle_diff), balcons, extras)
end

# The body, taking the core and handing it back, so the recipe and the eager
# core are the same model built two ways. `has_storage` is a BUILD-time fact,
# not data: storage adds seven variable blocks, so a recipe compiled for a
# storage case cannot instantiate one without it.
# `Nbus` here is a bus COUNT, not necessarily an `Int`: the recipe passes the
# deferred `length(data.bus)` and the eager path passes a number. Both flow
# through the index arithmetic below unchanged.
function build_mpopf_body(core, form, data, N, Nbus, user_callback, ::Type{T} = Float64;
                          storage_complementarity_constraint = false, has_storage = true) where {T}
    core, vars, cons = build_base_mpopf(core, form, data, N)
    core, vars, cons = add_mpopf_cons(core, form, data, N, Nbus, vars, cons, T)
    if has_storage
        core, vars, cons = build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)
        core, vars, cons =
            add_piecewise_cons(core, data, N, vars, cons, storage_complementarity_constraint, form)
    end
    core, vars2, cons2 = user_callback(core, vars, cons)
    return core, (; vars..., vars2...), (; cons..., cons2...)
end

"""
    mpopf_recipe(; N, form, has_storage, backend, T, user_callback)

The multi-period recipe. `N`, the load curve and whether the case has storage
are BUILD-time facts — the body slices `genarray[:, 2:N]` and storage changes
how many variable blocks there are — so a compiled library is per-(N, curve,
storage-shape). The network size is NOT: the bus axis is the extent of the
data, so one recipe instantiates any case. Close it with [`mpopf_args`](@ref).
"""
function mpopf_recipe(; N, has_storage = false, form::OPFForm = Polar(), backend = nothing,
                      T = Float64, user_callback = dummy_extension,
                      storage_complementarity_constraint = false)
    core, data = ExaCore(T; backend = backend, nargs = Val(1))
    return build_mpopf_body(core, form, data, N, length(data.bus), user_callback, T;
        storage_complementarity_constraint, has_storage)
end

function build_mpopf(data, Nbus, N, form, user_callback; backend = nothing, T = Float64, storage_complementarity_constraint = false, kwargs...)
    core = ExaCore(T; backend = backend)

    core, vars, cons = build_base_mpopf(core, form, data, N)
    core, vars, cons = add_mpopf_cons(core, form, data, N, Nbus, vars, cons, T)

    if length(data.storarray) > 0
        core, vars, cons = build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)
        core, vars, cons = add_piecewise_cons(core, data, N, vars, cons, storage_complementarity_constraint, form)
    end

    core, vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; prod = true, kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)
    return model, vars, cons
end

#different constraints used when a function is added to remove complementarity and make charge/discharge curve smooth
function build_mpopf(data, Nbus, N, discharge_func::Function, form, user_callback; backend = nothing, T = Float64, kwargs...)
    core = ExaCore(T; backend = backend)

    core, vars, cons = build_base_mpopf(core, form, data, N)
    core, vars, cons = add_mpopf_cons(core, form, data, N, Nbus, vars, cons, T)

    if length(data.storarray) > 0
        core, vars, cons = build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)
        core, vars, cons = add_smooth_cons(core, data, N, vars, cons, discharge_func, form)
    end

    core, vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; prod = true, kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)
    return model, vars, cons
end

# DC storage. The AC block models the CONVERTER — reactive injection `qint`,
# current magnitude `I2`, and the ohms relation tying them to the voltage —
# none of which the DC linearization has. The STORAGE itself is active power
# and carries over: `pst` into the bus, `E` as the state, `pstd`/`pstc` in and
# out. The `pst^2 + qst^2 <= rating^2` transfer limit collapses to a bound on
# `pst`, so it is a variable bound here rather than a constraint row.
function build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form::DC)
    @add_var(core, pst, length(data.storage), N;
        lvar = data.rep.nsrating, uvar = data.rep.srating)
    @add_var(core, E, length(data.storage), N;
        lvar = data.rep.zero_stor, uvar = data.rep.emax)
    @add_var(core, pstd, length(data.storage), N; uvar = data.rep.pdmax)
    vars = (; vars..., pst = pst, E = E, pstd = pstd)

    c_active_power_balance = cons.c_active_power_balance
    @add_con!(core, c_active_power_balance,
        s.storage_bus + Nbus*(t-1) => pst[s.i, t] for (s, t) in data.storarray)

    return core, vars, cons
end

function build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)

    #Storage specific variables

    #active/reactive power from bus into storage
    @add_var(core, pst, length(data.storage), N)
    @add_var(core, qst, length(data.storage), N)

    #current magnitude squared
    @add_var(core, I2, length(data.storage), N; lvar = data.rep.zero_stor)

    #ability of converter to control generation/absorption of reactive power
    @add_var(core, qint, length(data.storage), N; lvar = data.rep.nsrating, uvar = data.rep.srating)

    #energy/ state of charge
    @add_var(core, E, length(data.storage), N; lvar = data.rep.zero_stor, uvar = data.rep.emax)

    #discharge from battery to grid
    @add_var(core, pstd, length(data.storage), N; uvar = data.rep.pdmax)
    vars = (;vars..., pst=pst, qst=qst, I2=I2, qint=qint, E=E, pstd=pstd)

    c_active_power_balance = cons.c_active_power_balance
    c_reactive_power_balance = cons.c_reactive_power_balance

    @add_con!(core, c_active_power_balance, s.storage_bus + Nbus*(t-1) => pst[s.i, t] for (s, t) in data.storarray)
    @add_con!(core, c_reactive_power_balance, s.storage_bus + Nbus*(t-1) => qst[s.i, t] for (s, t) in data.storarray)

    @add_con(core, c_reactive_storage_power, c_reactive_stor_power(s, qst[s.i, t], qint[s.i, t], I2[s.i, t]) for (s, t) in data.storarray)

    @add_con(core, c_storage_transfer_thermal_limit, c_transfer_lim(s, pst[s.i, t], qst[s.i, t]) for (s, t) in data.storarray; lcon = data.rep.ninf_stor)

    if form isa Polar
        vm = vars.vm
        @add_con(core, c_ohms, c_ohms_polar(pst[s.i, t], qst[s.i, t], vm[s.storage_bus, t], I2[s.i, t]) for (s, t) in data.storarray)
    elseif form isa Rect
        vr = vars.vr
        vim = vars.vim
        @add_con(core, c_ohms, c_ohms_rect(pst[s.i, t], qst[s.i, t], vr[s.storage_bus, t], vim[s.storage_bus, t], I2[s.i, t]) for (s, t) in data.storarray)
    end

    cons = (;cons..., c_reactive_storage_power = c_reactive_storage_power, c_storage_transfer_thermal_limit = c_storage_transfer_thermal_limit, c_ohms=c_ohms)
    return core, vars, cons
end

function add_stor_power_con!(core, ::Union{Polar,Rect}, data, V, pstc)
    @add_con(core, c_active_storage_power,
        c_active_stor_power(s, V.pst[s.i, t], V.pstd[s.i, t], pstc[s.i, t], V.I2[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_con!(core, ::DC, data, V, pstc)
    @add_con(core, c_active_storage_power,
        c_active_stor_power_dc(s, V.pst[s.i, t], V.pstd[s.i, t], pstc[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_smooth_con!(core, ::Union{Polar,Rect}, data, V)
    @add_con(core, c_active_storage_power,
        c_active_storage_power_smooth(s, V.pst[s.i, t], V.pstd[s.i, t], V.I2[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_smooth_con!(core, ::DC, data, V)
    @add_con(core, c_active_storage_power,
        c_active_storage_power_smooth_dc(s, V.pst[s.i, t], V.pstd[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_piecewise_cons(core, data, N, vars, cons, storage_complementarity_constraint, form)
    #charge from battery to grid
    @add_var(core, pstc, length(data.storage), N; lvar = data.rep.zero_stor, uvar = data.rep.pcmax)
    vars = (;vars..., pstc=pstc)

    pst = vars.pst
    pstd = vars.pstd
    E = vars.E

    core, c_active_storage_power = add_stor_power_con!(core, form, data, vars, pstc)

    @add_con(core, c_storage_state, c_stor_state(s, E[s.i, t], E[s.i, t - 1], pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray_tail)

    @add_con(core, c_storage_state_init, c_stor_state(s, E[s.i, t], s.energy, pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray_head)

    @add_con(core, c_discharge_thermal_limit, c_discharge_lim(pstd[s.i, t], pstc[s.i, t]) for (s, t) in data.storarray; lcon = data.rep.nsrating, ucon = data.rep.srating)

    @add_con(core, c_discharge_positivity, pstd[s.i, t] for (s, t) in data.storarray; ucon = data.rep.inf_stor)

    #Complimentarity constraint
    if storage_complementarity_constraint
        @add_con(core, c_complementarity, c_comp(pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray)
        cons = (;cons..., c_complementarity = c_complementarity)
    end

    cons = (;cons...,
                c_active_storage_power = c_active_storage_power,
                c_storage_state = c_storage_state,
                c_storage_state_init = c_storage_state_init,
                c_discharge_thermal_limit = c_discharge_thermal_limit)

    return core, vars, cons
end

function add_smooth_cons(core, data, N, vars, cons, discharge_func, form)

    pst = vars.pst
    pstd = vars.pstd
    E = vars.E

    core, c_active_storage_power = add_stor_power_smooth_con!(core, form, data, vars)

    @add_con(core, c_storage_state, c_storage_state_smooth(s, E[s.i, t], E[s.i, t - 1], discharge_func, pstd[s.i, t]) for (s, t) in data.storarray_tail)

    @add_con(core, c_storage_state_init, c_storage_state_smooth(s, E[s.i, t], s.energy, discharge_func, pstd[s.i, t]) for (s, t) in data.storarray_head)

    @add_con(core, c_discharge_thermal_limit, c_discharge_limit_smooth(pstd[s.i, t]) for (s, t) in data.storarray; lcon = data.rep.nsrating, ucon = data.rep.srating)

    @add_con(core, c_charge_limit, pstd[s.i, t] for (s, t) in data.storarray; lcon = data.rep.npcmax, ucon = data.rep.inf_stor)

    cons = (;cons...,
                c_active_storage_power = c_active_storage_power,
                c_storage_state = c_storage_state,
                c_storage_state_init = c_storage_state_init,
                c_discharge_thermal_limit = c_discharge_thermal_limit,
                c_charge_limit = c_charge_limit)

    return core, vars, cons
end

"""
    mpopf_model(case, curve; kwargs...)
    mpopf_model(case, active_power_data, reactive_power_data; kwargs...)
    mpopf_model(case, curve, discharge_func::Function; kwargs...)
    mpopf_model(case, active_power_data, reactive_power_data, discharge_func::Function; kwargs...)

Construct a multi-period AC optimal power flow (MPOPF) model using different formats of load input data.

# Arguments

- `case`: the network, as for [`opf_model`](@ref): a path, a bare PGLib-OPF name, or a
  network already parsed with PowerIO.
- `curve::AbstractVector`: A time series of demand multiplier values. Each period's
  loads are the case's loads times `curve[t]`; bus shunts keep their value.
- `active_power_data::String`: Path to a matrix of active power loads (Pd) in MW, one
  row per bus of the model (isolated buses excluded) in file order and one column
  per period.
- `reactive_power_data::String`: Path to the matching matrix of reactive power loads (Qd).
- `discharge_func::Function`: (Optional) A function specifying battery discharge losses.

## Keyword Arguments

- `N::Int`: Number of time periods (inferred if not provided).
- `corrective_action_ratio::Float64`: Ratio of corrective power action allowed (default = 0.1).
- `backend`: Optimization solver backend (deault = nothing).
- `form::OPFForm`: the formulation — `Polar()` (default), `Rect()` or `DC()`.
- `T::Type`: Floating-point type for numeric variables (default = `Float64`).
- `storage_complementarity_constraint::Bool`: Whether to enforce complementarity for storage (only for some methods, default = false).
- `user_callback`: User function that extends the model
- `kwargs...`: Additional arguments passed to the solver or builder.

# Returns

A vector `(model::ExaModel object, variables::NamedTuple of variables, constraints::NamedTuple of constraints)` representing the MPOPF model.

# Method Variants

This function is overloaded for different combinations of input:

1. `mpopf_model(case, curve)`
2. `mpopf_model(case, active_power_data, reactive_power_data)`
3. `mpopf_model(case, curve, discharge_func)`
4. `mpopf_model(case, active_power_data, reactive_power_data, discharge_func)`
"""
function mpopf_model(
    case, curve;
    N = length(curve),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form::OPFForm = Polar(),
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)

    @assert length(curve) > 0
    data, = mpopf_args(case, curve; N, T, backend, corrective_action_ratio)
    Nbus = length(data.bus)

    return build_mpopf(data, Nbus, N, form, user_callback, backend = backend, T = T, storage_complementarity_constraint = storage_complementarity_constraint, kwargs...)

end

function mpopf_model(
    case, active_power_data, reactive_power_data;
    pd = readdlm(active_power_data),
    qd = readdlm(reactive_power_data),
    N = size(pd, 2),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form::OPFForm = Polar(),
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)

    d = parse_mp_power_data(case, N, corrective_action_ratio, T)
    update_load_data(d.busarray, pd, qd, d.baseMVA[])
    data = convert_data(_mp_narrow(d, N, T), backend)
    Nbus = length(data.bus)

    return build_mpopf(data, Nbus, N, form, user_callback, backend = backend, T = T, storage_complementarity_constraint = storage_complementarity_constraint, kwargs...)

end

#Input to discharge_func should be discharge rate (or negative charge), output should be loss in battery level
function mpopf_model(
    case, curve, discharge_func::Function;
    N = length(curve),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form::OPFForm = Polar(),
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)

    @assert length(curve) > 0
    data, = mpopf_args(case, curve; N, T, backend, corrective_action_ratio)
    Nbus = length(data.bus)

    return build_mpopf(data, Nbus, N, discharge_func, form,user_callback, backend = backend, T = T, kwargs...)

end

function mpopf_model(
    case, active_power_data, reactive_power_data, discharge_func::Function;
    pd = readdlm(active_power_data),
    qd = readdlm(reactive_power_data),
    N = size(pd, 2),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form::OPFForm = Polar(),
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)


    d = parse_mp_power_data(case, N, corrective_action_ratio, T)
    update_load_data(d.busarray, pd, qd, d.baseMVA[])
    data = convert_data(_mp_narrow(d, N, T), backend)
    Nbus = length(data.bus)

    return build_mpopf(data, Nbus, N, discharge_func, form,user_callback, backend = backend, T = T, kwargs...)
end

