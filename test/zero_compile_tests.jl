# The precompile workloads cover what a user does with a model: in a fresh Julia process,
# none of these compiles a method of ExaModelsPower at run time.

# Run `code` in a fresh process with this test environment and Julia's default flags
# (`Pkg.test` may run with bounds checking or coverage, for which there are no package
# images), and return the traced precompile statements of methods of `mod`. The code runs
# twice: the first run builds the package images, the second is the one traced.
function compiled_at_run_time(code, mod)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd(args...) = `$julia --startup-file=no --project=$(Base.active_project()) $args -e $code`
    run(cmd())
    trace = tempname()
    run(cmd("--trace-compile=$trace"))
    return filter(l -> occursin("typeof($mod.", l), readlines(trace))
end

const ZERO_COMPILE_SCRIPT = raw"""
using ExaModelsPower
import ExaModels: NLPModels
BACKEND
case = joinpath(pkgdir(ExaModelsPower), "data", "pglib_opf_case14_ieee.m")
for form in (Polar(), Rect(), DC())
    m, vars, cons = opf_model(case; form, kw...)
    m, vars, cons = opf_model(case; kw..., form)
    x = NLPModels.get_x0(m)
    NLPModels.obj(m, x)
    NLPModels.grad!(m, x, similar(x))
    NLPModels.cons!(m, x, similar(x, NLPModels.get_ncon(m)))
    xh = Array(x)
    y, z = zeros(NLPModels.get_ncon(m)), zeros(length(xh))
    write_matpower_solution(tempname() * ".m", case, vars, cons, xh, y, z, z;
                            form, objective = 0.0, status = "LOCALLY_SOLVED")
    opf_violations(m, x)
    case_start(case, m)
end
"""

function test_zero_compile(backend_code = "kw = (;)")
    code = replace(ZERO_COMPILE_SCRIPT, "BACKEND" => backend_code)
    compiled = compiled_at_run_time(code, "ExaModelsPower")
    isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
    @test isempty(compiled)
end
