# ── Security-constrained OPF, N-1 ───────────────────────────────────────────
#
# Two models of the same problem, in one file because they share their data and
# their physics:
#
#   * `scopf_model` — MONOLITHIC. The base OPF replicated across a scenario
#     axis: column 1 is the base case, columns 2..K+1 are the K contingencies.
#     This is the multi-period body with the time axis reinterpreted, so it is
#     built from `mpopf.jl`'s halves rather than from a second copy of them.
#   * `scopf_twostage_model` — the same problem written for MadNLP's
#     `SchurComplementCondensedKKTSystem`: the base case is the first stage and
#     each contingency is an `EachScenario()` block.
#
# A contingency trips either a generator (its post-contingency dispatch is
# fixed to zero) or a branch (masked out, so its flow is forced to zero and it
# drops out of the network). Post-contingency generation is the base dispatch
# plus a bounded corrective adjustment.
#
# All contingency information is baked into precomputed flat arrays so the
# generator expressions stay branch-free and the model stays generic over
# {T,VT} (CPU/GPU).
#
# `scopf_model` takes any `OPFForm` — `Polar()`, `Rect()` or `DC()`. A line
# outage is a mask, and the two formulations read two different fields: AC
# zeroes the admittance coefficients `c1..c8`, DC zeroes the susceptance `bs`
# that `_scopf_narrow` carries on each branch row. `scopf_twostage_model` is
# AC-only; `EachScenario()` is the whole point of it and a DC two-stage model
# has not been asked for.

# Reconstruct a branch keeping every field but replacing its thermal rating.
# Calls the all-fields inner constructor directly so the admittance
# coefficients are NOT recomputed.
function with_rate(b::ExaPowerIO.BranchData{T}, r) where {T}
    return ExaPowerIO.BranchData{T}(
        b.i, b.f_bus, b.t_bus, b.br_r, b.br_x, b.b_fr, b.b_to, b.g_fr, b.g_to,
        T(r), b.rate_b, b.rate_c, b.tap, b.shift, b.status, b.angmin, b.angmax,
        b.f_idx, b.t_idx,
        b.c1, b.c2, b.c3, b.c4, b.c5, b.c6, b.c7, b.c8,
    )
end

# Reconstruct a branch with its admittance coefficients zeroed (line outage).
# With c1..c8 = 0 the four flow equations force the branch's arc flows to zero,
# so it injects nothing into the power balance and its thermal limit
# (`-rate_a^2 <= 0`) is trivially slack.
function mask_branch(b::ExaPowerIO.BranchData{T}) where {T}
    return ExaPowerIO.BranchData{T}(
        b.i, b.f_bus, b.t_bus, b.br_r, b.br_x, b.b_fr, b.b_to, b.g_fr, b.g_to,
        b.rate_a, b.rate_b, b.rate_c, b.tap, b.shift, b.status, b.angmin, b.angmax,
        b.f_idx, b.t_idx,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
    )
end

# The parse plus the outage tables, shared by both models. `gen_out[k]` /
# `branch_out[k]` is what contingency `k` trips, 0 for neither; the monolithic
# model shifts them by one column, since its scenario 1 is the base case.
function _scopf_outages(filename, contingencies, unlimited_rate, ::Type{T}) where {T}
    d = parse_ac_power_data(filename, T)

    ngen = length(d.gen)
    nbranch = length(d.branch)
    Tr = eltype(d.rate_a)
    big = Tr(unlimited_rate)

    # MATPOWER convention: rateA == 0 means "unlimited". ExaPowerIO passes the 0
    # through, which would otherwise pin the corresponding flow variables to 0,
    # so replace zero ratings on both the branch field and the per-arc vector.
    branch = [iszero(b.rate_a) ? with_rate(b, big) : b for b in d.branch]
    rate_a = Tr[iszero(r) ? big : r for r in d.rate_a]

    K = length(contingencies)
    gen_out = zeros(Int, K)
    branch_out = zeros(Int, K)
    for (k, ct) in enumerate(contingencies)
        if ct.type == :gen
            (1 <= ct.idx <= ngen) ||
                error("generator contingency idx $(ct.idx) out of range 1:$ngen")
            gen_out[k] = ct.idx
        elseif ct.type == :branch
            (1 <= ct.idx <= nbranch) ||
                error("branch contingency idx $(ct.idx) out of range 1:$nbranch")
            branch_out[k] = ct.idx
        else
            error("unknown contingency type $(ct.type); use :gen or :branch")
        end
    end

    return (; d..., branch = branch, rate_a = rate_a), gen_out, branch_out
end

# ── the monolithic model: arguments ─────────────────────────────────────────

"""
    SCOPF_DEFAULT_CONTINGENCIES

The contingency list behind [`scopf_args_default`](@ref): the outage of branch
1 and of branch 2.

A compiled SCOPF library needs a ONE-argument, package-owned function to call —
the generated app resolves it by name from another process, so a closure over a
contingency list cannot be reached. `K` is a compile-time fact of such a library
anyway (see [`scopf_recipe`](@ref)), so the default lives here as a constant; a
different list means defining your own one-argument wrapper over
[`scopf_args`](@ref) in your own package.
"""
const SCOPF_DEFAULT_CONTINGENCIES =
    [(type = :branch, idx = 1), (type = :branch, idx = 2)]

"""
    scopf_args_default(filename) -> (data,)

[`scopf_args`](@ref) at [`SCOPF_DEFAULT_CONTINGENCIES`](@ref) — the spelling a
compiled library can call. Define your own one-argument wrapper in your package
for a different contingency list.
"""
scopf_args_default(filename) = scopf_args(
    filename, SCOPF_DEFAULT_CONTINGENCIES, length(SCOPF_DEFAULT_CONTINGENCIES),
    Float64, nothing, 0.05, 1.0e4,
)

"""
    scopf_args(filename, contingencies; K, T, backend, corrective_action_ratio,
               unlimited_rate) -> (data,)

The arguments that close [`scopf_recipe`](@ref): the parsed case with the
outages already applied, plus everything the recipe cannot compute from a
placeholder.

`contingencies` is NOT an argument to a compiled library — only `K` is baked
into the recipe, and the case file is the one value that crosses the boundary,
so a compiled SCOPF library is per-(K, contingency-list) and instantiates at any
network size.

The scenario-expanded bounds are grouped under `rep` rather than spread across
the top level, for the reason [`mpopf_args`](@ref) gives: `map` over a
NamedTuple stops inferring elementwise past 31 fields.
"""
scopf_args(filename, contingencies; K = length(contingencies), T = Float64,
           backend = nothing, corrective_action_ratio = 0.05,
           unlimited_rate = 1.0e4) =
    scopf_args(filename, contingencies, K, T, backend, corrective_action_ratio,
               unlimited_rate)

# The positional method is the one a compiled library reaches: `::Type{T}` makes
# `T` a static parameter, where the keyword form leaves it a `Type`-typed value
# nothing downstream can specialize on.
function scopf_args(filename, contingencies, K, ::Type{T}, backend,
                    corrective_action_ratio, unlimited_rate) where {T}
    # The recipe bakes K — it sizes every scenario axis — so a list of a
    # different length would build a model whose bounds and whose structure
    # disagree, silently.
    length(contingencies) == K || error(
        "scopf_args was given $(length(contingencies)) contingencies for K = $K",
    )
    d, gen_out, branch_out = _scopf_outages(filename, contingencies, unlimited_rate, T)
    return (convert_data(
        _scopf_narrow(d, gen_out, branch_out, K, corrective_action_ratio, T), backend),)
end

function _scopf_narrow(d, gen_out, branch_out, K, corrective_action_ratio, ::Type{T}) where {T}
    Ns = K + 1
    ngen = length(d.gen)

    # Scenario 1 is the base case, so the outage tables shift by one column.
    trip_gen = [c == 1 ? 0 : gen_out[c-1] for c in 1:Ns]
    trip_branch = [c == 1 ? 0 : branch_out[c-1] for c in 1:Ns]

    # The scenario axis is the second index, exactly as the time axis is in the
    # multi-period model — which is what lets that model's halves build this one.
    #
    # `bs` is the DC susceptance, and it carries the SAME outage the AC mask
    # carries: zero for the tripped branch in its own scenario, so the DC flow
    # equation degenerates to `pf == 0` and the branch drops out of the network.
    # It rides on the row because it cannot be derived from a masked `b` —
    # `mask_branch` zeroes the AC coefficients c1..c8, while the DC flow is
    # built from `br_r`/`br_x`, which a mask must leave alone (zeroing `br_x`
    # would give 0/0 wherever `br_r` is also zero, which is 3 of case9's 9
    # branches). So the two formulations mask two different fields of one row.
    barray = [
        trip_branch[c] == b.i ? (; b = mask_branch(b), c = c, bs = zero(T)) :
        (; b = b, c = c, bs = T(dc_susceptance(b)))
        for b in d.branch, c in 1:Ns
    ]
    busarray = [(; b, c) for b in d.bus, c in 1:Ns]
    arcarray = [(; a, c) for a in d.arc, c in 1:Ns]
    genarray = [(; g, c) for g in d.gen, c in 1:Ns]
    refarray = [(i, c) for i in d.ref_buses, c in 1:Ns]

    # Generation bounds per scenario; a tripped unit is fixed to zero in its own.
    pmin = repeat(d.pmin, 1, Ns); pmax = repeat(d.pmax, 1, Ns)
    qmin = repeat(d.qmin, 1, Ns); qmax = repeat(d.qmax, 1, Ns)

    # The corrective adjustment `extra >= 0`: zero in the base case, capped at a
    # fraction of pmax in a contingency, and zero for a unit that has tripped.
    extramin = zeros(T, ngen, Ns)
    extramax = zeros(T, ngen, Ns)
    for c = 2:Ns
        extramax[:, c] .= corrective_action_ratio .* d.pmax
    end

    for c = 2:Ns
        g = trip_gen[c]
        if g != 0
            pmin[g, c] = 0; pmax[g, c] = 0
            qmin[g, c] = 0; qmax[g, c] = 0
            extramax[g, c] = 0
        end
    end

    # Phase-angle-difference bounds; the outaged branch's limit is opened in its
    # own scenario, where the mask has left its terminals unrelated.
    angmin = repeat(d.angmin, 1, Ns); angmax = repeat(d.angmax, 1, Ns)
    for c = 2:Ns
        l = trip_branch[c]
        if l != 0
            angmin[l, c] = -Inf
            angmax[l, c] = Inf
        end
    end

    # Corrective coupling pg[g,c] = pg[g,1] + extra[g,c] for c >= 2, skipping a
    # unit that has tripped in that scenario — its dispatch is pinned to zero
    # and a coupling row would fix the base dispatch along with it.
    coupling = [(; g = d.gen[gi], c = c) for c = 2:Ns for gi = 1:ngen if trip_gen[c] != gi]

    # The DC flow variable is per BRANCH, where the AC ones are per arc, so it
    # needs its own rating vector. The static parser does not build one (the
    # static model builds it in `opf_args`), so it comes from the branch list
    # that `_scopf_outages` has already rate-fixed.
    branch_rate_a = T[br.rate_a for br in d.branch]

    # Named as the multi-period model names them, because its halves read them.
    rep = (;
        pmin, pmax, qmin, qmax,
        extramin, extramax,
        rate_a = repeat(d.rate_a, 1, Ns), nrate_a = repeat(-d.rate_a, 1, Ns),
        branch_rate_a = repeat(branch_rate_a, 1, Ns),
        nbranch_rate_a = repeat(-branch_rate_a, 1, Ns),
        vmin = repeat(d.vmin, 1, Ns), vmax = repeat(d.vmax, 1, Ns),
        vmin2 = repeat(d.vmin, 1, Ns) .^ 2, vmax2 = repeat(d.vmax, 1, Ns) .^ 2,
        angmin, angmax,
        ninf_b = fill(T(-Inf), size(barray)),
    )

    return (;
        bus = d.bus, gen = d.gen, arc = d.arc, branch = d.branch,
        refarray, barray, busarray, arcarray, genarray, coupling,
        rep,
    )
end

# ── the monolithic model: the body ──────────────────────────────────────────
#
# There is no SCOPF spine here. Replicating the base OPF over a second axis is
# what `mpopf.jl` already does, and every one of its halves reads the scenario
# axis as its second index — so this body is that one, with the ramp rate
# replaced by the corrective coupling and storage left out. `Nbus` is a bus
# COUNT, not necessarily an `Int`: the recipe passes the deferred
# `length(data.bus)` and the eager path a number.

function build_scopf_body(core, form::OPFForm, data, K, Nbus, user_callback,
                          ::Type{T} = Float64) where {T}
    Ns = K + 1

    core, G = add_gen_vars_mp!(core, form, data, Ns)
    @add_var(core, extra, length(data.gen), Ns;
        lvar = data.rep.extramin, uvar = data.rep.extramax)
    core, F = add_flow_vars_mp!(core, form, data, Ns)

    # Averaged over the base case and the contingencies, so the value is
    # comparable to a single-period cost.
    @add_obj(core, o, gen_cost(g, G.pg[g.i, 1]) for (g, c) in data.genarray)

    core, thermal = add_thermal_mp!(core, form, data, F)

    @add_con(core, c_corrective,
        G.pg[g.i, c] - G.pg[g.i, 1] - extra[g.i, c] for (g, c) in data.coupling)

    core, vars, cons = add_mpopf_cons(core, form, data, Ns, Nbus,
        merge(G, (; extra), F), merge(thermal, (; c_corrective)), T)

    core, vars2, cons2 = user_callback(core, vars, cons)
    return core, (; vars..., vars2...), (; cons..., cons2...)
end

"""
    scopf_recipe(; K, form, backend, T, user_callback) -> (core, variables, constraints)

The SCOPF *recipe* — an `ExaCore` holding the model's structure with its data
left open. Close it with [`scopf_args`](@ref).

`K`, the number of contingencies, is a BUILD-time fact: it sizes the scenario
axis of every variable block and of every expanded bound, and a count has no
symbolic form. WHICH contingencies is not — the outages are data, and travel in
the arguments — and neither is the network size, so one recipe at a given `K`
instantiates any case with any `K` outages.
"""
function scopf_recipe(; K, form::OPFForm = Polar(), backend = nothing,
                      T = Float64, user_callback = dummy_extension)
    core, data = ExaCore(T; backend = backend, nargs = Val(1))
    return build_scopf_body(core, form, data, K, length(data.bus), user_callback, T)
end

"""
    scopf_model(filename, contingencies; kwargs...)

Construct a hard-constrained N-1 security-constrained optimal power flow (SCOPF)
model.

The base OPF is replicated over scenarios: scenario 1 is the base case and each
entry of `contingencies` adds one post-contingency scenario. Post-contingency
active generation equals the base dispatch plus a non-negative corrective
adjustment bounded by `corrective_action_ratio * pmax`; a tripped unit is fixed
to zero. Power balance, branch flow equations and thermal limits, phase-angle
limits, and the reference-bus angle are enforced in every scenario, in whichever
formulation `form` names — the AC forms carry voltage and reactive power, `DC()`
carries neither.

A line outage is a mask: `Polar()`/`Rect()` zero the branch admittance
coefficients, `DC()` zeroes the branch susceptance. Either way the outaged
line's flow is forced to zero.

Defined as [`scopf_recipe`](@ref) instantiated at [`scopf_args`](@ref), so the
model solved here and the model compiled by `ExaModelsC` are built from one
definition rather than two.

# Arguments
- `filename::String`: Path to the network data file (MATPOWER `.m`).
- `contingencies::Vector{<:NamedTuple}`: each `(type = :gen, idx = i)`
  (generator outage) or `(type = :branch, idx = l)` (line outage). `idx` is the
  1-based index into the parsed `data.gen` / `data.branch` (matpower row order).

# Keyword Arguments
- `backend`: ExaModels backend (default `nothing`, CPU).
- `form`: the formulation, an [`OPFForm`](@ref) instance — `Polar()` (default),
  `Rect()` or `DC()`.
- `T::Type`: numeric type (default `Float64`).
- `corrective_action_ratio`: corrective redispatch cap as a fraction of `pmax`
  (default `0.05`).
- `unlimited_rate`: value (p.u.) substituted for a branch's `rate_a` when the
  data lists it as `0` (MATPOWER "unlimited"), so the flow is not pinned to zero
  (default `1.0e4`).
- `user_callback`: function `(core, vars, cons) -> (core, vars2, cons2)`
  extending the model.
- `kwargs...`: forwarded to `ExaModel`.

# Returns
`(model::ExaModel, vars::NamedTuple, cons::NamedTuple)`.
"""
function scopf_model(
    filename, contingencies;
    backend = nothing,
    form::OPFForm = Polar(),
    T = Float64,
    corrective_action_ratio = 0.05,
    unlimited_rate = 1.0e4,
    user_callback = dummy_extension,
    kwargs...,
)
    K = length(contingencies)
    core, vars, cons = scopf_recipe(;
        K = K, form = form, backend = backend, T = T, user_callback = user_callback)
    args = scopf_args(filename, contingencies;
        K = K, T = T, backend = backend,
        corrective_action_ratio = corrective_action_ratio,
        unlimited_rate = unlimited_rate)
    model = ExaModel(core, args...; prod = true, kwargs...)
    # The handles come out of the RECIPE, so their offsets are ArgNode
    # expressions; `solution(result, v)` cannot index with those. Resolve them
    # against the same arguments the model was built from.
    return model,
           ExaModels.instantiate(vars, args...),
           ExaModels.instantiate(cons, args...), args
end

# ── the two-stage model: arguments ──────────────────────────────────────────
#
# The same physics over a different partition. Here the base case is the DESIGN
# block and is NOT a scenario, so the scenario axis is K wide rather than K+1
# and the base-case data is read unexpanded — which is why this narrow carries
# both the flat fields and a `rep`.

function _scopf_twostage_narrow(d, gen_out, branch_out, K, corrective_action_ratio,
                                ::Type{T}) where {T}
    ngen = length(d.gen)
    nbranch = length(d.branch)

    barray_sc = [
        branch_out[k] == b.i ? (; b = mask_branch(b), k = k) : (; b = b, k = k)
        for b in d.branch, k = 1:K
    ]
    busarray_sc = [(; b, k) for b in d.bus, k = 1:K]
    arcarray_sc = [(; a, k) for a in d.arc, k = 1:K]
    genarray_sc = [(; g, k) for g in d.gen, k = 1:K]
    refarray_sc = [(i, k) for i in d.ref_buses, k = 1:K]

    # Uniform — every generator, every scenario. The monolithic model drops the
    # tripped unit's row instead; here the row is kept so that every scenario
    # block has the same shape, which is what the Schur partition factorizes.
    coupling_sc = [(; g, k) for g in d.gen, k = 1:K]

    pmin = repeat(d.pmin, 1, K); pmax = repeat(d.pmax, 1, K)
    qmin = repeat(d.qmin, 1, K); qmax = repeat(d.qmax, 1, K)
    extramin = zeros(T, ngen, K)
    extramax = repeat(corrective_action_ratio .* d.pmax, 1, K)
    for k = 1:K
        g = gen_out[k]
        if g != 0
            pmin[g, k] = 0; pmax[g, k] = 0
            qmin[g, k] = 0; qmax[g, k] = 0
            # Its coupling row is kept, so its `extra` must be able to absorb
            # `-pg` rather than constrain the base dispatch to zero.
            extramin[g, k] = -d.pmax[g]
            extramax[g, k] = 0
        end
    end

    angmin = repeat(d.angmin, 1, K); angmax = repeat(d.angmax, 1, K)
    for k = 1:K
        l = branch_out[k]
        if l != 0
            angmin[l, k] = -Inf
            angmax[l, k] = Inf
        end
    end

    rep = (;
        pmin, pmax, qmin, qmax,
        extramin, extramax,
        rate_a = repeat(d.rate_a, 1, K), nrate_a = repeat(-d.rate_a, 1, K),
        vmin = repeat(d.vmin, 1, K), vmax = repeat(d.vmax, 1, K),
        vmin2 = repeat(d.vmin, 1, K) .^ 2, vmax2 = repeat(d.vmax, 1, K) .^ 2,
        angmin, angmax,
        ninf_b = fill(T(-Inf), size(barray_sc)),
    )

    return (;
        # the design block reads these unexpanded — the static model's own shape,
        # which is what lets `add_balance!` and `add_extras!` build it
        bus = d.bus, gen = d.gen, arc = d.arc, branch = d.branch,
        ref_buses = d.ref_buses,
        vmin = d.vmin, vmax = d.vmax,
        pmin = d.pmin, pmax = d.pmax,
        qmin = d.qmin, qmax = d.qmax,
        angmin = d.angmin, angmax = d.angmax,
        rate_a = d.rate_a,
        vmin2 = d.vmin .^ 2, vmax2 = d.vmax .^ 2,
        branch_ninf = fill(T(-Inf), nbranch),
        barray_sc, busarray_sc, arcarray_sc, genarray_sc, refarray_sc, coupling_sc,
        rep,
    )
end

# NOT `scopf_twostage_args`. In this package `*_args` names the 1-tuple that
# closes a `*_recipe` under `nargs = Val(1)`; `scopf_args` and `mpopf_args` both
# end in `return (convert_data(...),)`. This returns `(data, K)`, and there is no
# two-stage recipe for it to close — `scopf_twostage_model` builds its core
# eagerly, needing a `TwoStageExaModelTag` at construction. So it is a private
# data-prep helper, and is named like one.
function _scopf_twostage_data(filename, contingencies, ::Type{T}, backend,
                              corrective_action_ratio, unlimited_rate) where {T}
    K = length(contingencies)
    K >= 1 || error(
        "scopf_twostage_model requires at least one contingency (K >= 1). " *
        "Use scopf_model for the no-contingency case.",
    )
    d, gen_out, branch_out = _scopf_outages(filename, contingencies, unlimited_rate, T)
    return convert_data(
        _scopf_twostage_narrow(d, gen_out, branch_out, K, corrective_action_ratio, T),
        backend), K
end

# ── the two-stage model: the per-stage halves ───────────────────────────────
#
# Variables only. Every EXPRESSION below is one of the shared selectors in
# opf.jl — `c_ref`, `ac_flow`, ... — called with no index tail for the design
# block and with the scenario index `k` as the tail for the scenario blocks, so
# the physics is written once for the whole package.
#
# Each of these returns TWO NamedTuples: one keyed as those shared selectors
# read — `va`, `pg`, `p` — and one keyed as a caller sees it, with the stage in
# the name. That is what lets a scenario block go through `ac_flow` unchanged
# while still coming back as `vak`/`pgk`.
#
# `Polar()` and `Rect()` only, unlike the monolithic model: `EachScenario()` is
# the only reason this half exists, and a DC two-stage model has not been asked
# for. The methods below are typed to say so rather than to fail deeper in.

function add_gen_vars_base!(core, ::Union{Polar,Rect}, data)
    @add_var(core, pg0, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg0, length(data.gen); lvar = data.qmin, uvar = data.qmax)
    return core, (; pg = pg0, qg = qg0), (; pg0, qg0)
end

function add_flow_vars_base!(core, ::Union{Polar,Rect}, data)
    @add_var(core, p0, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q0, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    return core, (; p = p0, q = q0), (; p0, q0)
end

function add_voltage_base!(core, ::Polar, data, ::Type{T}) where {T}
    @add_var(core, va0, length(data.bus); lvar = -pi, uvar = pi)
    @add_var(core, vm0, length(data.bus);
        start = one(T), lvar = data.vmin, uvar = data.vmax)
    return core, (; va = va0, vm = vm0), (; va0, vm0)
end

function add_voltage_base!(core, ::Rect, data, ::Type{T}) where {T}
    @add_var(core, vr0, length(data.bus); start = one(T))
    @add_var(core, vim0, length(data.bus);)
    return core, (; vr = vr0, vim = vim0), (; vr0, vim0)
end

function add_gen_vars_scen!(core, ::Union{Polar,Rect}, data)
    @add_var(core, pgk, EachScenario(), length(data.gen);
        lvar = data.rep.pmin, uvar = data.rep.pmax)
    @add_var(core, qgk, EachScenario(), length(data.gen);
        lvar = data.rep.qmin, uvar = data.rep.qmax)
    @add_var(core, extrak, EachScenario(), length(data.gen);
        lvar = data.rep.extramin, uvar = data.rep.extramax)
    return core, (; pg = pgk, qg = qgk, extra = extrak), (; pgk, qgk, extrak)
end

function add_flow_vars_scen!(core, ::Union{Polar,Rect}, data)
    @add_var(core, pk, EachScenario(), length(data.arc);
        lvar = data.rep.nrate_a, uvar = data.rep.rate_a)
    @add_var(core, qk, EachScenario(), length(data.arc);
        lvar = data.rep.nrate_a, uvar = data.rep.rate_a)
    return core, (; p = pk, q = qk), (; pk, qk)
end

function add_voltage_scen!(core, ::Polar, data, ::Type{T}) where {T}
    @add_var(core, vak, EachScenario(), length(data.bus); lvar = -pi, uvar = pi)
    @add_var(core, vmk, EachScenario(), length(data.bus);
        start = one(T), lvar = data.rep.vmin, uvar = data.rep.vmax)
    return core, (; va = vak, vm = vmk), (; vak, vmk)
end

function add_voltage_scen!(core, ::Rect, data, ::Type{T}) where {T}
    @add_var(core, vrk, EachScenario(), length(data.bus); start = one(T))
    @add_var(core, vimk, EachScenario(), length(data.bus);)
    return core, (; vr = vrk, vim = vimk), (; vrk, vimk)
end

# The rectangular form's extra rows, per stage. Polar has none.
add_vmag_scen!(core, ::Polar, data, V) = (core, (;))
function add_vmag_scen!(core, ::Rect, data, V)
    @add_con(core, c_voltage_magnitude_k, EachScenario(),
        c_voltage_magnitude_rect(V.vr[b.i, k], V.vim[b.i, k]) for (b, k) in data.busarray_sc;
        lcon = data.rep.vmin2, ucon = data.rep.vmax2)
    return core, (; c_voltage_magnitude_k)
end

# The design block: the static AC OPF, built from the static model's own halves.
# Its objective is scaled with the scenarios' so the total is the average cost.
function add_scopf_design!(core, form, data, K, V, G, F)
    @add_obj(core, o_base, gen_cost(g, G.pg[g.i]) for g in data.gen)
    @add_con(core, c_ref_angle, c_ref(form, V, i) for i in data.ref_buses)
    core, flowcons = add_flow_constraints!(core, form, data, V, F)
    @add_con(core, c_phase_angle_diff, c_angle(form, b, V) for b in data.branch;
        lcon = data.angmin, ucon = data.angmax)
    core, balcons = add_balance!(core, form, data, V, G, F)
    core, extras = add_extras!(core, form, data, V, F)
    return core, merge((; c_ref_angle), flowcons, (; c_phase_angle_diff), balcons, extras)
end

# A scenario block: the same model at `[i, k]`, every row tagged `EachScenario()`.
function add_scopf_scenario!(core, form, data, K, Nbus, V, G, F)

    @add_con(core, c_ref_angle_k, EachScenario(),
        c_ref(form, V, i, k) for (i, k) in data.refarray_sc)
    @add_con(core, c_to_active_power_flow_k, EachScenario(),
        ac_flow(form, :ta, b, F, V, k) for (b, k) in data.barray_sc)
    @add_con(core, c_to_reactive_power_flow_k, EachScenario(),
        ac_flow(form, :tr, b, F, V, k) for (b, k) in data.barray_sc)
    @add_con(core, c_from_active_power_flow_k, EachScenario(),
        ac_flow(form, :fa, b, F, V, k) for (b, k) in data.barray_sc)
    @add_con(core, c_from_reactive_power_flow_k, EachScenario(),
        ac_flow(form, :fr, b, F, V, k) for (b, k) in data.barray_sc)
    @add_con(core, c_phase_angle_diff_k, EachScenario(),
        c_angle(form, b, V, k) for (b, k) in data.barray_sc;
        lcon = data.rep.angmin, ucon = data.rep.angmax)

    @add_con(core, c_active_power_balance_k, EachScenario(),
        c_bal_p(form, b, V, k) for (b, k) in data.busarray_sc)
    @add_con(core, c_reactive_power_balance_k, EachScenario(),
        c_bal_q(form, b, V, k) for (b, k) in data.busarray_sc)

    core, vmag = add_vmag_scen!(core, form, data, V)

    # The balance ROWS and the appends into them are separate steps for the
    # reason `add_balance_cons_mp!` gives: the rectangular form inserts its
    # voltage-magnitude rows between them, and appending earlier would put the
    # COO triplets in a different order.
    @add_con!(core, c_active_power_balance_k,
        a.bus + Nbus * (k - 1) => F.p[a.i, k] for (a, k) in data.arcarray_sc)
    @add_con!(core, c_reactive_power_balance_k,
        a.bus + Nbus * (k - 1) => F.q[a.i, k] for (a, k) in data.arcarray_sc)
    @add_con!(core, c_active_power_balance_k,
        g.bus + Nbus * (k - 1) => -G.pg[g.i, k] for (g, k) in data.genarray_sc)
    @add_con!(core, c_reactive_power_balance_k,
        g.bus + Nbus * (k - 1) => -G.qg[g.i, k] for (g, k) in data.genarray_sc)

    @add_con(core, c_from_thermal_limit_k, EachScenario(),
        c_thermal_limit(b, F.p[b.f_idx, k], F.q[b.f_idx, k]) for (b, k) in data.barray_sc;
        lcon = data.rep.ninf_b)
    @add_con(core, c_to_thermal_limit_k, EachScenario(),
        c_thermal_limit(b, F.p[b.t_idx, k], F.q[b.t_idx, k]) for (b, k) in data.barray_sc;
        lcon = data.rep.ninf_b)

    return core, merge(
        (; c_ref_angle_k, c_to_active_power_flow_k, c_to_reactive_power_flow_k,
           c_from_active_power_flow_k, c_from_reactive_power_flow_k,
           c_phase_angle_diff_k, c_active_power_balance_k, c_reactive_power_balance_k),
        vmag,
        (; c_from_thermal_limit_k, c_to_thermal_limit_k))
end

"""
    scopf_twostage_model(filename, contingencies; backend, T, form,
                         corrective_action_ratio, unlimited_rate, user_callback, kwargs...)

Two-stage form of the hard-constrained N-1 SCOPF ([`scopf_model`](@ref)), built
on `TwoStageExaCore` from ExaModels.jl so it can be solved with MadNLP's
`SchurComplementCondensedKKTSystem`, which factorizes the per-contingency block
structure in parallel.

Same physics as [`scopf_model`](@ref) — the expressions are the package's
shared ones in both — but a different partition:

- **First stage (design):** the base-case OPF — generation, the form's voltage
  variables, arc flows, and their power-balance / flow / limit constraints.
- **Second stage (`EachScenario`):** for each contingency `k`, the
  post-contingency OPF and its constraints, plus the coupling
  `pgk - pg0 - extrak = 0` (the only rows that touch both stages).

Requires `K = length(contingencies) >= 1`. `form` is `Polar()` or `Rect()`: this
model is AC-only, where [`scopf_model`](@ref) also takes `DC()`.

# Returns
`(model, vars, cons, post_solve_info)` where
`post_solve_info = (; ns, nv, nd, nc, nc_design, var_scen, con_scen)` gives the
Schur dimensions (scenarios / vars-per-scenario / design vars / cons-per-scenario
/ design-only constraints) plus the per-variable and per-constraint scenario
tags. Pass the tags through `kkt_options` so the Schur solver partitions by tag
(design and scenario variables are interleaved here, not contiguous):

```julia
using MadNLP, MadNLPGPU, CUDA, CUDSS
model, vars, cons, info = scopf_twostage_model(case, conts; backend = CUDABackend())
result = madnlp(model;
    kkt_system    = SchurComplementCondensedKKTSystem,
    linear_solver = MadNLPGPU.CUDSSSolver,
    kkt_options   = Dict(:schur_ns => info.ns, :schur_nv => info.nv,
                         :schur_nd => info.nd, :schur_nc => info.nc,
                         :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen),
)
```

!!! note "Design-only base-case constraints"
    The base-case physics are first-stage (design) constraints — `nc_design` of
    them, reported in `post_solve_info`. `SchurComplementCondensedKKTSystem`
    folds the design equalities into the bordered first-stage block and
    condenses the design inequalities into it, so this two-stage model and the
    monolithic [`scopf_model`](@ref) converge to the same optimum. The base case
    produces these rows by construction — they are deliberately NOT worked
    around by replicating base physics into every scenario.
"""
function scopf_twostage_model(
    filename, contingencies;
    backend = nothing,
    T = Float64,
    form::Union{Polar,Rect} = Polar(),
    corrective_action_ratio = 0.05,
    unlimited_rate = 1.0e4,
    user_callback = dummy_extension,
    kwargs...,
)
    data, K = _scopf_twostage_data(filename, contingencies, T, backend,
                                   corrective_action_ratio, unlimited_rate)

    # The underlying ExaCore is built explicitly so `T` is honored —
    # `TwoStageExaCore()` would force Float64.
    core = ExaCore(T;
        backend = backend,
        tag = ExaModels.TwoStageExaModelTag(
            K,
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend),
        ),
    )
    Nbus = length(data.bus)

    core, V0, V0pub = add_voltage_base!(core, form, data, T)
    core, G0, G0pub = add_gen_vars_base!(core, form, data)
    core, F0, F0pub = add_flow_vars_base!(core, form, data)

    core, Vk, Vkpub = add_voltage_scen!(core, form, data, T)
    core, Gk, Gkpub = add_gen_vars_scen!(core, form, data)
    core, Fk, Fkpub = add_flow_vars_scen!(core, form, data)

    core, cons0 = add_scopf_design!(core, form, data, K, V0, G0, F0)
    core, consk = add_scopf_scenario!(core, form, data, K, Nbus, Vk, Gk, Fk)

    # The only rows touching both stages.
    @add_con(core, c_corrective, EachScenario(),
        Gk.pg[g.i, k] - G0.pg[g.i] - Gk.extra[g.i, k] for (g, k) in data.coupling_sc)

    vars = merge(V0pub, G0pub, F0pub, Vkpub, Gkpub, Fkpub)
    cons = merge(cons0, consk, (; c_corrective))

    # Schur dimensions AND the full per-variable/per-constraint scenario tags.
    # MadNLP's SchurComplementCondensedKKTSystem partitions by these tags
    # (design and scenario variables are interleaved here, not contiguous), and
    # folds the design-only base-case constraints into its first-stage bordered
    # block. Pass them through `kkt_options` as :schur_var_scen / :schur_con_scen.
    var_scen = Vector{Int}(Array(core.tag.var_scen))
    con_scen = Vector{Int}(Array(core.tag.con_scen))
    post_solve_info = (
        ns = K,
        nv = count(==(1), var_scen),
        nd = count(==(0), var_scen),
        nc = count(==(1), con_scen),
        nc_design = count(==(0), con_scen),
        var_scen = var_scen,
        con_scen = con_scen,
    )

    core, vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; kwargs...)

    return model, (; vars..., vars2...), (; cons..., cons2...), post_solve_info
end
