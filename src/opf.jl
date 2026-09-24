function dummy_extension(core, vars, cons)
    return core, (;), (;)
end

# ── Recipe, arguments, model ──────────────────────────────────────────────────
#
# The AC OPF is written once, as a *recipe*: an `ExaCore` whose structure is
# fixed but whose data is left open, standing in as an `ExaModels.ArgSource`
# placeholder.  `opf_args` produces the data that closes it, and
# `ac_opf_model` is the two composed — so there is one definition of the model
# rather than two that can drift.
#
# The reason for the split is ahead-of-time compilation.  Writing a model
# against placeholders *is* what `ExaModelsC` needs in order to compile it into
# a shared library with the data left open; the author never has to reason
# about what `juliac --trim=safe` can digest.
#
# ── What the placeholder cannot do, the caller supplies ──
#
# A placeholder supports field access and deferred arithmetic — `data.bus`,
# `length(data.gen)`, `-data.rate_a` — and nothing else.  An array built to a
# deferred size, a broadcast, or a comprehension has no symbolic form, and
# `ExaModels` refuses them rather than silently mis-building.  Three things in
# this model were written that way before, and none of them is dropped: each
# becomes a value `opf_args` computes and passes in as ordinary data.
#
#   * starting points, previously `fill!(similar(data.bus, T), one(T))`;
#   * the rectangular form's squared voltage bounds, previously `data.vmin.^2`;
#   * the thermal limits' `-Inf` lower bounds, previously a filled array.
#
# `opf_args` is the adapter between what a *user* has — a case file, and
# preferences about how to start — and what the recipe (and, past it, the
# ExaModelsC boundary) can carry, which is data and only data.

"""
    opf_start(value, n, T)

Materialize a user-supplied starting point into a length-`n` `Vector{T}`.

`value` may be
- a scalar — the flat start, e.g. `one(T)` for voltage magnitude;
- an `AbstractVector` of length `n`;
- a `Base.Generator`, e.g. `Base.Generator(f, 1:n)` — note that `f` should be a
  *named* function rather than an anonymous one, so that the type survives
  ahead-of-time compilation and reads intelligibly in errors.

Generators and vectors are collected here, in `opf_args`, rather than being
referred to from the recipe: only data crosses into an instantiated model, so
resolving them at this point is what lets the same start work for an in-Julia
model and for a compiled library.
"""
opf_start(v::Number, n, ::Type{T}) where {T} = fill(T(v), n)
opf_start(g::Base.Generator, n, ::Type{T}) where {T} = _opf_start_check(collect(T, g), n)
opf_start(v::AbstractVector, n, ::Type{T}) where {T} =
    _opf_start_check(convert(Vector{T}, collect(v)), n)

function _opf_start_check(v, n)
    length(v) == n || throw(
        DimensionMismatch(
            "a starting point of length $(length(v)) was given for a variable " *
            "block of length $n",
        ),
    )
    return v
end

# The defaults are the ones this model has always used: a flat start of 1.0 for
# voltage magnitude, zero everywhere else (`zero(T)` is also `ExaModels`' own
# default for a variable with no `start`).
_default_start(::Type{T}) where {T} =
    (va = zero(T), vm = one(T), vr = one(T), vim = zero(T),
     pg = zero(T), qg = zero(T), p = zero(T), q = zero(T), pf = zero(T))

# A start entry may also be given as a function of the parsed data, which is how
# a caller warm-starts from the case's own operating point — `start = (; vm = d
# -> d.vm0, va = d -> d.va0)`.  Resolved before materialization, so the result
# is still ordinary data.
_resolve_start(f, data) = f(data)
_resolve_start(v::Union{Number,AbstractVector,Base.Generator}, data) = v

"""
    opf_args(filename; T = Float64, backend = nothing, start = (;))
        -> (data,)

Return the argument tuple that closes [`opf_recipe`](@ref) — the parsed case
together with everything the recipe cannot compute from a placeholder.

`start` overrides any of the starting points `va`, `vm`, `vr`, `vim`, `pg`,
`qg`, `p`, `q`. Each entry may be a scalar, a vector, a `Base.Generator`, or a
function of the parsed data (`d -> d.vm0` to warm-start from the case's own
operating point). Defaults are a flat `1.0` voltage magnitude and zero
elsewhere, which is what this model used before the recipe split.

# Example
```julia
args = opf_args("pglib_opf_case118_ieee.m"; start = (vm = d -> d.vm0, va = d -> d.va0))
model = ExaModel(ac_opf_recipe()[1], args...)
```
"""
opf_args(filename; T = Float64, backend = nothing, start = (;)) =
    opf_args(filename, T, backend, start)

# The positional method is the one a compiled library calls: `::Type{T}` makes
# `T` a static parameter, where the keyword form leaves it a `Type`-typed value
# that nothing downstream can specialize on.
function opf_args(filename, ::Type{T}, backend = nothing, start = (;)) where {T}
    p = parse_ac_power_data(filename, T)
    s = merge(_default_start(T), NamedTuple(start))
    bus, gen, arc, branch = p.bus, p.gen, p.arc, p.branch
    nbus, ngen, narc, nbranch = length(bus), length(gen), length(arc), length(branch)

    # EXACTLY the fields the model bodies read, and no more. `map` over a
    # NamedTuple — which is what `convert_data` is — stays inferable up to 31
    # fields and gives up at 32 (measured, this Julia); passing the parse
    # through whole came to 35, and the result was a `NamedTuple` with concrete
    # names and no value types at all. That one abstract value was enough to
    # leave `ExaModel(CORE, data)` unresolved under `--trim=safe`.
    #
    # Nothing is lost by narrowing: the fields dropped here — `baseMVA`,
    # `storage`, `vm0`/`va0`/`pg0`/`qg0` and the storage ratings — are inputs to
    # the starting points, which are resolved just below and travel as the
    # `*_start` arrays.
    data = (;
        bus, gen, arc, branch,
        ref_buses = p.ref_buses,
        vmin = p.vmin, vmax = p.vmax,
        pmin = p.pmin, pmax = p.pmax,
        qmin = p.qmin, qmax = p.qmax,
        angmin = p.angmin, angmax = p.angmax,
        rate_a = p.rate_a,
        # Per BRANCH, not per arc. `dcopf`'s `pf` is a branch-length block and
        # was bounded by the arc-length `rate_a`; that lands on the right
        # values only because the first `nbranch` arcs happen to be the
        # from-arcs in branch order (measured identical, 0 differences, on
        # case14, case118 and case9241_pegase). Correct by construction rather
        # than by that coincidence.
        branch_rate_a = [br.rate_a for br in p.branch],
        # Broadcasting a placeholder is refused by design, so the rectangular
        # form's squared voltage bounds are computed here, beside the parse.
        vmin2 = p.vmin .^ 2,
        vmax2 = p.vmax .^ 2,
        # The thermal limits are one-sided; the bound array used to be built
        # with `fill!(similar(...), -Inf)` against a length only known later.
        branch_ninf = fill(T(-Inf), nbranch),
        # Resolved against the FULL parse, so `start = (vm = d -> d.vm0,)`
        # still reaches a field the model itself never sees.
        va_start = opf_start(_resolve_start(s.va, p), nbus, T),
        vm_start = opf_start(_resolve_start(s.vm, p), nbus, T),
        vr_start = opf_start(_resolve_start(s.vr, p), nbus, T),
        vim_start = opf_start(_resolve_start(s.vim, p), nbus, T),
        pg_start = opf_start(_resolve_start(s.pg, p), ngen, T),
        qg_start = opf_start(_resolve_start(s.qg, p), ngen, T),
        p_start = opf_start(_resolve_start(s.p, p), narc, T),
        q_start = opf_start(_resolve_start(s.q, p), narc, T),
        pf_start = opf_start(_resolve_start(s.pf, p), nbranch, T),
    )

    return (convert_data(data, backend),)
end

# ── One body, three formulations ─────────────────────────────────────────────
#
# Polar, rectangular and DC all follow the same spine:
#
#   variables → objective → ref angle → flows → angle diff → balance → extras
#
# so there is one body and a method per formulation for the steps that differ.
# The order above is the order each formulation already used, which is what
# lets the merge be exact: every model built here is identical to the one its
# own builder produced, COO order included.
#
# The body takes the core and hands it back — `@add_var` and `@add_con` rebind
# their first argument, so a caller keeping its own binding is left with the
# blocks and stale counters, which fails much later at instantiation with
# `Nonsensical dimensions`. `data` is either an `ArgSource` placeholder (a
# recipe) or a concrete NamedTuple (an eager core); the body cannot tell.

"""
    OPFForm

The formulation a model is built in: [`Polar`](@ref), [`Rect`](@ref) or
[`DC`](@ref). Passed as an instance — `opf_model(file; form = Rect())` — so the
choice is part of the TYPE the compiler sees. (Symbols were the pre-0.4
spelling; they made every call through them type-unstable and are gone.)
"""
abstract type OPFForm end

"Polar-voltage AC: variables `va`, `vm`."
struct Polar <: OPFForm end

"Rectangular-voltage AC: variables `vr`, `vim`, plus the voltage-magnitude rows."
struct Rect <: OPFForm end

"The DC linearization: `va` only, flows per branch, no reactive half."
struct DC <: OPFForm end



# ── variables ────────────────────────────────────────────────────────────────

function add_voltage!(core, ::Polar, d)
    @add_var(core, va, length(d.bus); start = d.va_start)
    @add_var(core, vm, length(d.bus); start = d.vm_start, lvar = d.vmin, uvar = d.vmax)
    return core, (; va, vm)
end

function add_voltage!(core, ::Rect, d)
    @add_var(core, vr, length(d.bus); start = d.vr_start)
    @add_var(core, vim, length(d.bus); start = d.vim_start)
    return core, (; vr, vim)
end

function add_generation!(core, ::Union{Polar,Rect}, d)
    @add_var(core, pg, length(d.gen); start = d.pg_start, lvar = d.pmin, uvar = d.pmax)
    @add_var(core, qg, length(d.gen); start = d.qg_start, lvar = d.qmin, uvar = d.qmax)
    return core, (; pg, qg)
end

function add_flows!(core, ::Union{Polar,Rect}, d)
    @add_var(core, p, length(d.arc); start = d.p_start, lvar = -d.rate_a, uvar = d.rate_a)
    @add_var(core, q, length(d.arc); start = d.q_start, lvar = -d.rate_a, uvar = d.rate_a)
    return core, (; p, q)
end

# ── the expressions that differ by formulation ──────────────────────────────
#
# Every selector carries a trailing `idx...`, so ONE definition serves the
# static model (empty tail, `V.va[i]`) and every model that replicates the base
# problem over a second axis — the multi-period one at `[i, t]` and the
# security-constrained one at `[i, c]`. They were three copies of this family
# before; the copies differed only by the index tail and by the `_mp` suffix, so
# nothing here is a generalization of the static form, it IS the static form.
#
# The splat is resolved at compile time (the tail's length is in the type), so
# the traced expression is identical to the hand-written index — `V.va[i]` when
# the tail is empty, `V.va[i, t]` when it is `(t,)`.

@inline c_ref(::Union{Polar,DC}, V, i, idx...) = c_ref_angle_polar(V.va[i, idx...])
@inline c_ref(::Rect, V, i, idx...) = c_ref_angle_rect(V.vr[i, idx...], V.vim[i, idx...])

@inline c_angle(::Union{Polar,DC}, b, V, idx...) =
    c_phase_angle_diff_polar(b, V.va[b.f_bus, idx...], V.va[b.t_bus, idx...])
@inline c_angle(::Rect, b, V, idx...) =
    c_phase_angle_diff_rect(b, V.vr[b.f_bus, idx...], V.vr[b.t_bus, idx...],
        V.vim[b.f_bus, idx...], V.vim[b.t_bus, idx...])

@inline c_bal_p(::Polar, b, V, idx...) = c_active_power_balance_demand_polar(b, V.vm[b.i, idx...])
@inline c_bal_p(::Rect, b, V, idx...) =
    c_active_power_balance_demand_rect(b, V.vr[b.i, idx...], V.vim[b.i, idx...])
# The DC active balance takes no voltage; `V` and the index tail are carried for
# uniformity with the other formulations.
@inline c_bal_p(::DC, b, V, idx...) = c_active_power_balance_dc(b)
@inline c_bal_q(::Polar, b, V, idx...) = c_reactive_power_balance_demand_polar(b, V.vm[b.i, idx...])
@inline c_bal_q(::Rect, b, V, idx...) =
    c_reactive_power_balance_demand_rect(b, V.vr[b.i, idx...], V.vim[b.i, idx...])

# The DC branch flow. `bs` is passed rather than read off `br` so a replicated
# model can hand in a MASKED susceptance and so express a line outage; see
# `dc_susceptance` in constraint.jl.
@inline dc_ohms(br, bs, F, V, idx...) =
    c_ohms_law_dcopf(bs, F.pf[br.i, idx...], V.va[br.f_bus, idx...], V.va[br.t_bus, idx...])

# The four AC branch flows, selected by a `Val` rather than by four more helper
# names per formulation. This is the one place the merged form reads worse than
# the duplicated one; it is eight one-line methods against eight inline
# expressions, and it keeps `add_flow_constraints!` shared.
@inline ac_flow(f, s::Symbol, b, F, V, idx...) = ac_flow(f, Val(s), b, F, V, idx...)
@inline ac_flow(::Polar, ::Val{:ta}, b, F, V, idx...) =
    c_to_active_power_flow_polar(b, F.p[b.f_idx, idx...], V.vm[b.f_bus, idx...], V.vm[b.t_bus, idx...], V.va[b.f_bus, idx...], V.va[b.t_bus, idx...])
@inline ac_flow(::Polar, ::Val{:tr}, b, F, V, idx...) =
    c_to_reactive_power_flow_polar(b, F.q[b.f_idx, idx...], V.vm[b.f_bus, idx...], V.vm[b.t_bus, idx...], V.va[b.f_bus, idx...], V.va[b.t_bus, idx...])
@inline ac_flow(::Polar, ::Val{:fa}, b, F, V, idx...) =
    c_from_active_power_flow_polar(b, F.p[b.t_idx, idx...], V.vm[b.f_bus, idx...], V.vm[b.t_bus, idx...], V.va[b.f_bus, idx...], V.va[b.t_bus, idx...])
@inline ac_flow(::Polar, ::Val{:fr}, b, F, V, idx...) =
    c_from_reactive_power_flow_polar(b, F.q[b.t_idx, idx...], V.vm[b.f_bus, idx...], V.vm[b.t_bus, idx...], V.va[b.f_bus, idx...], V.va[b.t_bus, idx...])
@inline ac_flow(::Rect, ::Val{:ta}, b, F, V, idx...) =
    c_to_active_power_flow_rect(b, F.p[b.f_idx, idx...], V.vr[b.f_bus, idx...], V.vr[b.t_bus, idx...], V.vim[b.f_bus, idx...], V.vim[b.t_bus, idx...])
@inline ac_flow(::Rect, ::Val{:tr}, b, F, V, idx...) =
    c_to_reactive_power_flow_rect(b, F.q[b.f_idx, idx...], V.vr[b.f_bus, idx...], V.vr[b.t_bus, idx...], V.vim[b.f_bus, idx...], V.vim[b.t_bus, idx...])
@inline ac_flow(::Rect, ::Val{:fa}, b, F, V, idx...) =
    c_from_active_power_flow_rect(b, F.p[b.t_idx, idx...], V.vr[b.f_bus, idx...], V.vr[b.t_bus, idx...], V.vim[b.f_bus, idx...], V.vim[b.t_bus, idx...])
@inline ac_flow(::Rect, ::Val{:fr}, b, F, V, idx...) =
    c_from_reactive_power_flow_rect(b, F.q[b.t_idx, idx...], V.vr[b.f_bus, idx...], V.vr[b.t_bus, idx...], V.vim[b.f_bus, idx...], V.vim[b.t_bus, idx...])

# ── constraint groups ───────────────────────────────────────────────────────

function add_flow_constraints!(core, form::Union{Polar,Rect}, d, V, F)
    @add_con(core, c_to_active_power_flow, ac_flow(form, :ta, b, F, V) for b in d.branch)
    @add_con(core, c_to_reactive_power_flow, ac_flow(form, :tr, b, F, V) for b in d.branch)
    @add_con(core, c_from_active_power_flow, ac_flow(form, :fa, b, F, V) for b in d.branch)
    @add_con(core, c_from_reactive_power_flow, ac_flow(form, :fr, b, F, V) for b in d.branch)
    return core,
    (;
        c_to_active_power_flow,
        c_to_reactive_power_flow,
        c_from_active_power_flow,
        c_from_reactive_power_flow,
    )
end

function add_balance!(core, form::Union{Polar,Rect}, d, V, G, F)
    @add_con(core, c_active_power_balance, c_bal_p(form, b, V) for b in d.bus)
    @add_con(core, c_reactive_power_balance, c_bal_q(form, b, V) for b in d.bus)
    @add_con!(core, c_active_power_balance, a.bus => F.p[a.i] for a in d.arc)
    @add_con!(core, c_reactive_power_balance, a.bus => F.q[a.i] for a in d.arc)
    @add_con!(core, c_active_power_balance, g.bus => -G.pg[g.i] for g in d.gen)
    @add_con!(core, c_reactive_power_balance, g.bus => -G.qg[g.i] for g in d.gen)
    return core, (; c_active_power_balance, c_reactive_power_balance)
end

function add_extras!(core, ::Polar, d, V, F)
    @add_con(core, c_from_thermal_limit,
        c_thermal_limit(b, F.p[b.f_idx], F.q[b.f_idx]) for b in d.branch; lcon = d.branch_ninf)
    @add_con(core, c_to_thermal_limit,
        c_thermal_limit(b, F.p[b.t_idx], F.q[b.t_idx]) for b in d.branch; lcon = d.branch_ninf)
    return core, (; c_from_thermal_limit, c_to_thermal_limit)
end

function add_extras!(core, ::Rect, d, V, F)
    @add_con(core, c_from_thermal_limit,
        c_thermal_limit(b, F.p[b.f_idx], F.q[b.f_idx]) for b in d.branch; lcon = d.branch_ninf)
    @add_con(core, c_to_thermal_limit,
        c_thermal_limit(b, F.p[b.t_idx], F.q[b.t_idx]) for b in d.branch; lcon = d.branch_ninf)
    @add_con(core, c_voltage_magnitude,
        c_voltage_magnitude_rect(V.vr[b.i], V.vim[b.i]) for b in d.bus;
        lcon = d.vmin2, ucon = d.vmax2)
    return core, (; c_from_thermal_limit, c_to_thermal_limit, c_voltage_magnitude)
end

# ── the spine ───────────────────────────────────────────────────────────────

function build_opf(core, form::OPFForm, data, user_callback, ::Type{T}) where {T}
    core, V = add_voltage!(core, form, data)
    core, G = add_generation!(core, form, data)
    core, F = add_flows!(core, form, data)

    @add_obj(core, o, gen_cost(g, G.pg[g.i]) for g in data.gen)
    @add_con(core, c_ref_angle, c_ref(form, V, i) for i in data.ref_buses)
    core, flowcons = add_flow_constraints!(core, form, data, V, F)
    @add_con(core, c_phase_angle_diff, c_angle(form, b, V) for b in data.branch;
        lcon = data.angmin, ucon = data.angmax)
    core, balcons = add_balance!(core, form, data, V, G, F)
    core, extras = add_extras!(core, form, data, V, F)

    vars = merge(V, G, F)
    cons = merge((; c_ref_angle), flowcons, (; c_phase_angle_diff), balcons, extras)
    core, vars2, cons2 = user_callback(core, vars, cons)
    return core, (; vars..., vars2...), (; cons..., cons2...)
end

"""
    ac_opf_recipe(; backend, T, form, user_callback) -> (core, variables, constraints)

Return the AC OPF *recipe* — an `ExaCore` holding the model's structure with its
data left open — together with the variable and constraint handles.

Close it with [`opf_args`](@ref):

```julia
core, vars, cons = ac_opf_recipe()
model = ExaModel(core, opf_args("pglib_opf_case118_ieee.m")...)
```

The same recipe instantiates at any case; it is not consumed by the first use.

# Arguments
- `backend`: the array backend to build against. Default `nothing` (CPU).
- `T`: the numeric type (default `Float64`).
- `form`: the formulation, an [`OPFForm`](@ref) instance — `Polar()` (default), `Rect()` or `DC()`.
- `user_callback`: user function that extends the model.
"""
function opf_recipe(;
    backend = nothing,
    T = Float64,
    form::OPFForm = Polar(),
    user_callback = dummy_extension,
)
    core, data = ExaCore(T; backend = backend, nargs = Val(1))
    return build_opf(core, form, data, user_callback, T)
end

"""
    ac_opf_model(filename; backend, T, form, user_callback, start, kwargs...)

Return `ExaModel`, variables, and constraints for a static AC Optimal Power Flow
(ACOPF) problem from the given file.

Defined as [`opf_recipe`](@ref) instantiated at [`opf_args`](@ref), so the
model solved here and the model compiled by `ExaModelsC` are built from one
definition rather than two.

# Arguments
- `filename::String`: Path to the data file.
- `backend`: The solver backend to use. Default if nothing.
- `T`: The numeric type to use (default is `Float64`).
- `form`: the formulation, an [`OPFForm`](@ref) instance — `Polar()` (default), `Rect()` or `DC()`.
- `user_callback`: User function that extends the model
- `start`: Starting-point overrides; see [`opf_args`](@ref).
- `kwargs...`: Additional keyword arguments passed to the model builder.

# Returns
A vector `(model, variables, constraints)`:
- `model`: An `ExaModel` object.
- `variables`: NamedTuple of model variables.
- `constraints`: NamedTuple of model constraints.
"""
function opf_model(
    filename;
    backend = nothing,
    T = Float64,
    form::OPFForm = Polar(),
    user_callback = dummy_extension,
    start = (;),
    kwargs...,
)
    core, vars, cons =
        opf_recipe(; backend = backend, T = T, form = form, user_callback = user_callback)
    args = opf_args(filename; T = T, backend = backend, start = start)
    model = ExaModel(core, args...; prod = true, kwargs...)
    # The handles come out of the RECIPE, so their offsets are ArgNode
    # expressions; `solution(result, v)` cannot index with those. Resolve them
    # against the same arguments the model was built from. This was lost once
    # already, in the entry-point unification — test_solution_handles is the
    # regression test that catches it.
    return model,
           ExaModels.instantiate(vars, args...),
           ExaModels.instantiate(cons, args...)
end

# Takes the core and hands it back: `@add_var` and `@add_con` rebind their
# first argument, so a caller keeping its own binding is left with the blocks
# and stale counters. `data` is either an `ArgSource` placeholder (the recipe)
# or a concrete NamedTuple (the eager core) — one body serves both.
# ── DC: the methods that differ from the AC spine ───────────────────────────
#
# The body is `build_opf` in opf.jl; only these differ. DC has no reactive
# half, its flows are per branch rather than per arc, and it has no thermal
# limits — so `add_extras!` is where the merge would start costing more than it
# saves if a formulation needed several such no-ops. It needs one.

function add_voltage!(core, ::DC, d)
    @add_var(core, va, length(d.bus); start = d.va_start)
    return core, (; va)
end

function add_generation!(core, ::DC, d)
    @add_var(core, pg, length(d.gen); start = d.pg_start, lvar = d.pmin, uvar = d.pmax)
    return core, (; pg)
end

function add_flows!(core, ::DC, d)
    @add_var(core, pf, length(d.branch); start = d.pf_start,
        lvar = -d.branch_rate_a, uvar = d.branch_rate_a)
    return core, (; pf)
end

# The DC selectors are the Polar ones: `c_ref`, `c_angle` and `c_bal_p` are
# defined once, above, with `DC` in their signatures.

function add_flow_constraints!(core, ::DC, d, V, F)
    # The susceptance is a per-branch constant here — the static model has no
    # outage to mask it for — so it is computed from the branch in place.
    @add_con(core, c_ohms_law,
        dc_ohms(br, dc_susceptance(br), F, V) for br in d.branch)
    return core, (; c_ohms_law)
end

function add_balance!(core, form::DC, d, V, G, F)
    @add_con(core, c_active_power_balance, c_bal_p(form, b, V) for b in d.bus)
    @add_con!(core, c_active_power_balance, g.bus => -G.pg[g.i] for g in d.gen)
    @add_con!(core, c_active_power_balance, br.f_bus => F.pf[br.i] for br in d.branch)
    @add_con!(core, c_active_power_balance, br.t_bus => -F.pf[br.i] for br in d.branch)
    return core, (; c_active_power_balance)
end

add_extras!(core, ::DC, d, V, F) = (core, (;))

# ── DC entry points ─────────────────────────────────────────────────────────
#
# The body, the arguments and the three entry points are shared with AC; these
# are the DC-named spellings of them, kept because `dcopf_model(file)` reads
# better than `opf_model(file; form = :dc)` at a call site that only ever wants
# DC.

dcopf_recipe(; kwargs...) = opf_recipe(; form = DC(), kwargs...)
dcopf_model(filename; kwargs...) = opf_model(filename; form = DC(), kwargs...)

# ── The old names ───────────────────────────────────────────────────────────
#
# `ac_opf_*` and `dcopf_*` are one family now — the body is the same and the
# formulation is an argument — so these are the shared entry points under the
# names callers already use. `DC()` is a formulation like any other, so
# `ac_opf_model(f; form = DC())` is legal and means what it says; `dcopf_model`
# is the spelling that says it in the name.

ac_opf_recipe(; form = Polar(), kwargs...) = opf_recipe(; form = form, kwargs...)
ac_opf_model(filename; form = Polar(), kwargs...) = opf_model(filename; form = form, kwargs...)
ac_opf_args(filename; kwargs...) = opf_args(filename; kwargs...)
