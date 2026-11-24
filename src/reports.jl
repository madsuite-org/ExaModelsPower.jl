# Text about an OPF instance and a solution of it: the reports a compiled library carries
# (cnlp.h, the reports tier), which `cnlp solve` prints before and after a solve.

_nblock(model, name) = hasfield(typeof(model.refs), name) ? ExaModels.total(getfield(model.refs, name).size) : 0
_sig(v) = string(round(Float64(v); sigdigits = 3))

"""
    opf_report_instance(case_file, model)

A line about an OPF instance: the case it was built from, its buses, generators and
branches, and its size. The signature of a report (`compile_all`'s `reports`).
"""
function opf_report_instance(case_file::AbstractString, model::ExaModels.ExaModel)
    nbus = _nblock(model, :va)
    ngen = _nblock(model, :pg)
    narc = max(_nblock(model, :p), _nblock(model, :pf))
    return string(
        basename(case_file), ": ", nbus, " buses, ", ngen, " generators, ",
        _nblock(model, :pf) > 0 ? narc : narc ÷ 2, " branches; ",
        model.meta.nvar, " variables, ", model.meta.ncon, " constraints",
    )
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
    io = IOBuffer()
    print(io, "generation cost ", _sig(objective), "; largest violation by group:")
    for (k, name) in enumerate(keys(v))
        print(io, "\n  ", rpad(replace(String(name), '_' => ' '), 18), _sig(v[k]))
    end
    return String(take!(io))
end
