# The largest violation of each group of an OPF model's constraints and bounds at a point:
# power balance, branch flow limits, angle differences, voltage and generation bounds.

# The blocks of each group, by the names `opf_model` gives them; a formulation has only
# some of them (`vm` is a polar variable, `c_voltage_magnitude` a rectangular constraint).
const VIOLATION_GROUPS = (
    power_balance = (:c_active_power_balance, :c_reactive_power_balance),
    power_flow = (:c_to_active_power_flow, :c_to_reactive_power_flow, :c_from_active_power_flow,
                  :c_from_reactive_power_flow, :c_ohms_law, :c_ref_angle),
    branch_flow = (:c_from_thermal_limit, :c_to_thermal_limit, :p, :q, :pf),
    angle_difference = (:c_phase_angle_diff,),
    voltage = (:vm, :c_voltage_magnitude),
    generation = (:pg, :qg),
)

"""
    opf_violations(model, x)

The largest absolute violation of each group of constraints and bounds of `model`, an
OPF model built by `opf_model` (any form, on any backend), at the point `x`, as a
`NamedTuple` of `Float64`s in the model's units (per unit, radians):

- `power_balance`: active and reactive power balance at each bus;
- `power_flow`: the branch flow equations (Ohm's law in `DC()`) and the reference angle;
- `branch_flow`: the branch flow limits, `p² + q² ≤ rate²` and the flow bounds;
- `angle_difference`: the phase angle difference limits;
- `voltage`: the voltage magnitude bounds (a variable bound in `Polar()`, a
  constraint in `Rect()`; none in `DC()`);
- `generation`: the active and reactive generation bounds.

The violation of `l ≤ v ≤ u` is `max(l - v, v - u, 0)`; a group a formulation does not
have is `0`. Constraint values are evaluated at `x`.
"""
function opf_violations(model::ExaModels.ExaModel, x)
    c = similar(x, model.meta.ncon)
    ExaModels.cons!(model, x, c)
    m = model.meta
    return _violations(model.refs, Array(x), Array(m.lvar), Array(m.uvar), Array(c), Array(m.lcon), Array(m.ucon))
end

function _violations(r, x, lvar, uvar, c, lcon, ucon)
    g(names) = _group_violation(r, names, x, lvar, uvar, c, lcon, ucon)
    G = VIOLATION_GROUPS
    return (power_balance = g(G.power_balance), power_flow = g(G.power_flow),
            branch_flow = g(G.branch_flow), angle_difference = g(G.angle_difference),
            voltage = g(G.voltage), generation = g(G.generation))
end

function _group_violation(r::NamedTuple, names, x, lvar, uvar, c, lcon, ucon)
    v = 0.0
    for name in names
        hasfield(typeof(r), name) || continue
        b = getfield(r, name)
        v = max(v, _block_violation(b, x, lvar, uvar, c, lcon, ucon))
    end
    return v
end

_block_violation(b::ExaModels.AbstractVariable, x, lvar, uvar, c, lcon, ucon) = _range_violation(b, x, lvar, uvar)
_block_violation(b::ExaModels.AbstractConstraint, x, lvar, uvar, c, lcon, ucon) = _range_violation(b, c, lcon, ucon)

function _range_violation(b, v, lo, hi)
    w = 0.0
    for i in (b.offset + 1):(b.offset + ExaModels.total(b.size))
        w = max(w, Float64(lo[i] - v[i]), Float64(v[i] - hi[i]))
    end
    return w
end

"""
    write_violations(out, case_file, model, x, y, zl, zu; objective = NaN, status = "")

Write [`opf_violations`](@ref)`(model, x)` to `out` as CSV, one `group,violation` row per
group. The signature of a solution writer, so that a compiled library carries it; see
`compile_all`. `case_file`, the multipliers, `objective` and `status` are not used.
"""
function write_violations(out::AbstractString, case_file::AbstractString, model::ExaModels.ExaModel,
                          x, y, zl, zu; objective = NaN, status::AbstractString = "")
    v = opf_violations(model, x)
    io = open(out, "w")
    try
        println(io, "group,violation")
        for (k, name) in enumerate(keys(v))
            println(io, name, ",", v[k])
        end
    finally
        close(io)
    end
    return out
end
