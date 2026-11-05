# The precompile workloads cover what a user does with a model: in a fresh Julia process,
# none of these compiles anything of ExaModelsPower at run time.

# Run `code` in a fresh process with this test environment and Julia's default flags, and
# return the traced precompile statements that mention `modname`, and whether the trace
# holds the planted `zero_compile_probe` (a positive control: the trace was written and the
# filter still matches its format). The code runs twice: the first run builds the package
# images, the second is the one traced. `Pkg.test` precompiles for its own flags (bounds
# checking, coverage), so on a cold depot the first run precompiles the script's
# dependencies again, for the default flags: about a minute on the CPU, several with CUDA.
function compiled_at_run_time(code, modname)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    code = code * "\nzero_compile_probe(x) = x + 1; zero_compile_probe(1)\n"
    cmd(args...) = `$julia --startup-file=no --project=$(Base.active_project()) $args -e $code`
    run(cmd())
    trace = tempname()
    run(cmd("--trace-compile=$trace"))
    lines = readlines(trace)
    rm(trace)
    return filter(l -> occursin("$modname.", l), lines),
           any(l -> occursin("typeof(Main.zero_compile_probe)", l), lines)
end

const ZERO_COMPILE_SCRIPT = raw"""
using ExaModelsPower
import ExaModels: NLPModels
BACKEND
case = joinpath(pkgdir(ExaModelsPower), "data", "pglib_opf_case14_ieee.m")
mktempdir() do dir
    for form in (Polar(), Rect(), DC())
        m, vars, cons = opf_model(case; form, kw...)
        m, vars, cons = opf_model(case; kw..., form)
        x = NLPModels.get_x0(m)
        NLPModels.obj(m, x)
        NLPModels.grad!(m, x, similar(x))
        NLPModels.cons!(m, x, similar(x, NLPModels.get_ncon(m)))
        xh = Array(x)
        y, z = zeros(eltype(xh), NLPModels.get_ncon(m)), zeros(eltype(xh), length(xh))
        write_matpower_solution(joinpath(dir, "solution.m"), case, vars, cons, xh, y, z, z;
                                form, objective = 0.0, status = "LOCALLY_SOLVED")
        opf_violations(m, x)
        case_start(case, m)
    end
end
"""

function test_zero_compile(backend_code = "kw = (;)")
    code = replace(ZERO_COMPILE_SCRIPT, "BACKEND" => backend_code)
    compiled, probe = compiled_at_run_time(code, "ExaModelsPower")
    @test probe
    isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
    @test isempty(compiled)
end
