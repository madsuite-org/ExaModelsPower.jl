# Text about an OPF instance and a solution of it: the reports a compiled library carries
# (cnlp.h, the reports tier), which `cnlp solve` prints before and after a solve. Written to
# compile under `juliac --trim=safe`: blocks are looked up by constant names, and the text
# is built by concatenating strings.

# the size of block `name` of the model, 0 when its formulation has none
_nblock(model, ::Val{name}) where {name} =
    hasfield(typeof(model.refs), name) ? ExaModels.total(getfield(model.refs, name).size)::Int : 0
_sig(v, d = 3) = string(round(Float64(v); sigdigits = d))

"""
    opf_report_instance(case_file, model)

A line about an OPF instance: the case it was built from, its buses, generators and
branches, and its size. The signature of a report (`compile_all`'s `reports`).
"""
function opf_report_instance(case_file::AbstractString, model::ExaModels.ExaModel)
    nbus = _nblock(model, Val(:va))
    ngen = _nblock(model, Val(:pg))
    npf = _nblock(model, Val(:pf))               # DC: one flow per branch
    nbranch = npf > 0 ? npf : _nblock(model, Val(:p)) ÷ 2   # AC: one per arc, two arcs a branch
    return basename(String(case_file)) * ": " * string(nbus) * " buses, " * string(ngen) *
        " generators, " * string(nbranch) * " branches; " * string(model.meta.nvar) *
        " variables, " * string(model.meta.ncon) * " constraints"
end

"""
    opf_report_solution(case_file, model, x, y, zl, zu; objective = NaN, status = "")

A summary of a solution: the generation cost (the objective) and the largest violation of
each group of constraints and bounds ([`opf_violations`](@ref)), in per unit and radians.
The signature of a report (`compile_all`'s `reports`).
"""
function opf_report_solution(case_file::AbstractString, model::ExaModels.ExaModel, x, y, zl, zu;
                             objective = NaN, status::AbstractString = "")
    v = opf_violations(model, x)
    return "generation cost " * _sig(objective, 10) * "; largest violation by group:" *
        "\n  power balance     " * _sig(v.power_balance) *
        "\n  power flow        " * _sig(v.power_flow) *
        "\n  branch flow       " * _sig(v.branch_flow) *
        "\n  angle difference  " * _sig(v.angle_difference) *
        "\n  voltage           " * _sig(v.voltage) *
        "\n  generation        " * _sig(v.generation)
end
