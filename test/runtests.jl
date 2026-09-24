using Test, ExaModelsPower, MadNLP, MadNLPGPU, KernelAbstractions, CUDA, CUDSS, PowerModels, Ipopt, JuMP, ExaModels, NLPModelsJuMP
# Import only `CNLPModel`, not all of CNLPModels: it exports `solution` and so
# does ExaModels, and a name exported by two loaded packages resolves to
# neither -- which took out 24 tests that call `solution` unqualified.
using ExaModelsCompiler
using CNLPModels: CNLPModel
import NLPModels
import LinearAlgebra

include("opf_tests.jl")
include("recipe_tests.jl")
include("scopf_tests.jl")

# CI runs each backend, and the GOC3 smoke test, as a separate job, so the wall clock is
# the slowest of them rather than their sum (they were 13.0, 14.8, 21.5 and 74.9 min in run
# 30814411004).  EMP_TEST_SELECTION names the slice; it defaults to everything, so a plain
# local `Pkg.test()` still runs the whole suite.
const SELECTION = get(ENV, "EMP_TEST_SELECTION", "all")
const VALID_SELECTIONS = ("all", "nothing", "cpu", "cuda", "goc3", "aot")
SELECTION in VALID_SELECTIONS ||
    error("EMP_TEST_SELECTION must be one of $(join(VALID_SELECTIONS, ", ")), got $(repr(SELECTION))")

# Asking for the GPU slice on a machine with no GPU must fail loudly: silently leaving
# CONFIGS empty would make the job pass without running a single test.
if SELECTION == "cuda" && !CUDA.has_cuda_gpu()
    error("EMP_TEST_SELECTION=cuda but no CUDA device is visible")
end

const CONFIGS = Any[]
SELECTION in ("all", "nothing") && push!(CONFIGS, nothing)
SELECTION in ("all", "cpu") && push!(CONFIGS, CPU())
SELECTION in ("all", "cuda") && CUDA.has_cuda_gpu() && push!(CONFIGS, CUDABackend())
const RUN_GOC3 = SELECTION in ("all", "goc3")
# Its own slice, like GOC3, rather than an opt-in env var: `compile_all` is red
# at main and nothing says so, because the gate it sat behind is set by no CI
# job. A slice shows up in the job list; an env var can be forgotten.
const RUN_AOT = SELECTION in ("all", "aot")

isempty(CONFIGS) && !RUN_GOC3 && !RUN_AOT &&
    error("EMP_TEST_SELECTION=$(SELECTION) selected no tests")

test_cases = [("../data/pglib_opf_case3_lmbd.m", "case3", test_case3),
              ("../data/pglib_opf_case5_pjm.m", "case5", test_case5),
              ("../data/pglib_opf_case14_ieee.m", "case14", test_case14)]

#MP
#MP solutions hard coded based on solutions computer 4/10/2025 on CPU with 1e-8 tol
#Curve = [1, .9, .8, .95, 1]
true_sol_case3_curve = 25384.366465
true_sol_case3_pregen = 29049.351564
true_sol_case5_curve = 78491.04247
true_sol_case5_pregen = 87816.396884
#W storage
true_sol_case3_curve_stor = 25358.8275
true_sol_case3_curve_stor_func = 25352.57 
true_sol_case3_pregen_stor = 29023.691
true_sol_case3_pregen_stor_func = 29019.32 
true_sol_case5_curve_stor = 68782.0125
true_sol_case5_curve_stor_func = 69271.9 
true_sol_case5_pregen_stor = 79640.085
true_sol_case5_pregen_stor_func = 79630.4 
mp_test_cases = [("../data/pglib_opf_case3_lmbd.m", "case3", "../data/case3_5split.Pd", "../data/case3_5split.Qd", true_sol_case3_curve, true_sol_case3_pregen),
                 ("../data/pglib_opf_case5_pjm.m", "case5", "../data/case5_5split.Pd", "../data/case5_5split.Qd", true_sol_case5_curve, true_sol_case5_pregen)]

mp_stor_test_cases = [("../data/pglib_opf_case3_lmbd_mod.m", "case3", "../data/case3_5split.Pd", "../data/case3_5split.Qd",
                        true_sol_case3_curve_stor, true_sol_case3_curve_stor_func, true_sol_case3_pregen_stor, true_sol_case3_pregen_stor_func),
                        ("../data/pglib_opf_case5_pjm_mod.m", "case5", "../data/case5_5split.Pd", "../data/case5_5split.Qd",
                        true_sol_case5_curve_stor, true_sol_case5_curve_stor_func, true_sol_case5_pregen_stor, true_sol_case5_pregen_stor_func)]

# The SCOPF recipe/eager check runs over each kind of contingency list, since a
# generator outage changes the model's STRUCTURE (its coupling row is dropped)
# where a line outage only changes data. Indices valid in every `test_cases` case.
scopf_recipe_contingencies = [
    ("branch", ExaModelsPower.SCOPF_DEFAULT_CONTINGENCIES),
    ("gen", [(type = :gen, idx = 1), (type = :gen, idx = 2)]),
    ("gen+branch", [(type = :gen, idx = 1), (type = :branch, idx = 1),
                    (type = :gen, idx = 2), (type = :branch, idx = 2)]),
]

static_forms = [("rect", Rect(), ACRPowerModel, test_rect_voltage),
                ("polar", Polar(), ACPPowerModel, test_polar_voltage)]

mp_forms = [("rect", Rect()), ("polar", Polar())]

function example_func(d, srating)
    return d + 20/srating*d^2
end

untimed_elec_data = [
(i = 1, bus = 1, cost = -50);
(i = 2, bus = 2, cost = -20)
]
Ntime = 3
Nbus = 2
elec_data = [(;b..., t = t) for b in untimed_elec_data, t in 1:Ntime]
elec_min = [0, 0]
elec_max = [50, 50]
elec_scale = 5
elec_curve = [1, .9, .95]

function add_electrolyzers(core, vars, cons)
    @add_var(core, p_elec, size(elec_data, 1),
    size(elec_data, 2); lvar = elec_min, uvar = elec_max)
    @add_obj(core, o2,
    e.cost*p_elec[e.i, e.t] for e in elec_data)
    @add_con!(core, cons.c_active_power_balance,
    e.bus + Nbus*(e.t-1) => p_elec[e.i, e.t]
    for e in elec_data)
    @add_con(core, c_elec_ramp,
    p_elec[e.i, e.t] - p_elec[e.i, e.t - 1]
    for e in elec_data[:, 2:Ntime];
    lcon = fill!(similar(elec_data, Float64,
    length(elec_data)), -elec_scale),
    ucon = fill!(similar(elec_data, Float64,
    length(elec_data)), elec_scale))
    vars = (p_elec=p_elec,)
    cons = (c_elec_ramp=c_elec_ramp,)
    return core, vars, cons
end

PowerModels.silence()

function parse_pm(filename)
    data = PowerModels.parse_file(filename)
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)

    return data
end

function runtests()
    @testset "ExaModelsPower test" begin

        # Solving is the expensive half of this suite, and MadNLP is not what this package
        # is responsible for: its job is to build correct models.  So the smallest case is
        # solved once on the CPU and once on the GPU -- static AC against the
        # PowerModels/Ipopt reference, and multi-period against the hardcoded objective --
        # and every other configuration is checked by evaluating its callbacks.
        solve_here(backend) = backend === nothing || backend isa CUDABackend
        solve_case, solve_form = "case3", "rect"

        # The PowerModels/Ipopt and JuMP/MadNLP reference solutions do not depend on the
        # backend, so compute them once per case/form and reuse them.
        static_ref_cache = Dict{Tuple{String,String},Any}()

        # case3 and case5 exercise the same code with different data, so the multi-period
        # sections keep case3 only.  Every backend runs the same set: with one job per
        # backend the wall clock is the slowest backend, not the sum of all of them.
        mp_cases = mp_test_cases[1:1]
        mp_stor  = mp_stor_test_cases[1:1]

        # The recipe split: `ac_opf_model` is `ac_opf_recipe` instantiated at
        # `ac_opf_args`. It must agree with the same body built eagerly, in
        # BOTH formulations —
        # that is the whole guarantee the split is for, and it is backend-free,
        # so it runs once rather than per backend.
        if SELECTION in ("all", "nothing")
            for (filename, case, _) in test_cases, (form_str, form, _, _) in static_forms
                @testset "$case, recipe == eager, $form_str" begin
                    test_recipe_equivalence(filename, form)
                end
                for (ctg_str, ctgs) in scopf_recipe_contingencies
                    @testset "$case, SCOPF recipe == eager, $form_str, $ctg_str" begin
                        test_scopf_recipe_equivalence(filename, form; contingencies = ctgs)
                    end
                end
                @testset "$case, solution handles, $form_str" begin
                    test_solution_handles(filename, form)
                end
            end

            # DC is not in `static_forms` -- those entries carry a PowerModels
            # type and a voltage test, and DC has neither -- so the DC SCOPF's
            # recipe/eager equivalence is checked here instead. Its line outage
            # is a different mask from the AC one (`bs`, not `c1..c8`), so it
            # gets the same guarantee rather than inheriting the AC result.
            for (filename, case, _) in test_cases, (ctg_str, ctgs) in scopf_recipe_contingencies
                @testset "$case, SCOPF recipe == eager, dc, $ctg_str" begin
                    test_scopf_recipe_equivalence(filename, DC(); contingencies = ctgs)
                end
            end

        end

        # Once for the whole suite, not once per case and formulation:
        # `compile_all` is minutes of juliac. Backend-free, so it is its own
        # slice rather than part of the `nothing` one.
        if RUN_AOT
            @testset "compile_all, then the models it returns" begin
                test_aot()
            end
        end

        for backend in CONFIGS
            solving = solve_here(backend)

            # Static AC
            for (filename, case, test_function) in test_cases
                for (form_str, form, power_model, test_voltage) in static_forms
                    m32, _, _ = ac_opf_model(filename; T=Float32, backend = backend, form=form)
                    m64, v64, _ = ac_opf_model(filename; T=Float64, backend = backend, form=form)

                    @testset "$case, static, $backend, $form_str" begin
                        test_callbacks(m32, m64, backend)
                    end

                    if solving && case == solve_case && form_str == solve_form
                        va64, vm64, pg64, qg64, p64, q64 = v64
                        result64 = exasolve(m64, backend; print_level = MadNLP.ERROR)

                        result_pm, result_nlp_pm = get!(static_ref_cache, (filename, form_str)) do
                            nlp_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>Float64(result64.options.tol), "print_level"=>0)
                            rpm = solve_opf(filename, power_model, nlp_solver)

                            m_pm = JuMP.Model()
                            instantiate_model(parse_pm(filename), power_model, PowerModels.build_opf, jump_model = m_pm)
                            rnlp = madnlp(MathOptNLPModel(m_pm); print_level = MadNLP.ERROR)
                            (rpm, rnlp)
                        end

                        @testset "$case, static solve, $backend, $form_str" begin
                            test_function(result64, result_pm, result_nlp_pm, pg64, qg64, p64, q64)
                            test_voltage(result64, result_pm, va64, vm64)
                        end
                    end
                end
            end

            # Multi-period
            for (form_str, symbol) in mp_forms
                for (filename, case, Pd_pregen, Qd_pregen, true_sol_curve, true_sol_pregen) in mp_cases
                    variants = [
                        ("curve",        () -> (T -> mpopf_model(filename, [1, .9, .8, .95, 1]; T = T, backend = backend, form = symbol))),
                        ("curve, func",  () -> (T -> mpopf_model(filename, [1, .9, .8, .95, 1], example_func; T = T, backend = backend, form = symbol))),
                        ("pregen",       () -> (T -> mpopf_model(filename, Pd_pregen, Qd_pregen; T = T, backend = backend, form = symbol))),
                        ("pregen, func", () -> (T -> mpopf_model(filename, Pd_pregen, Qd_pregen, example_func; T = T, backend = backend, form = symbol))),
                    ]
                    for (label, mk) in variants
                        build = mk()
                        m32, _, _ = build(Float32)
                        m64, _, _ = build(Float64)

                        @testset "$(case), MP, $(backend), $(label), $(form_str)" begin
                            test_callbacks(m32, m64, backend)
                        end

                        # The one solve kept for the objective regression: the hardcoded
                        # multi-period solutions are the only check on mpopf answers.
                        if solving && case == solve_case && form_str == solve_form && label == "curve"
                            result64 = exasolve(m64, backend; print_level = MadNLP.ERROR)
                            @testset "$(case), MP solve, $(backend), $(label), $(form_str)" begin
                                test_mp_case(result64, true_sol_curve)
                            end
                        end
                    end
                end

                # Multi-period with storage
                for (filename, case, Pd_pregen, Qd_pregen, true_sol_curve_stor,
                    true_sol_curve_stor_func, true_sol_pregen_stor, true_sol_pregen_stor_func) in mp_stor

                    variants = [
                        ("curve",        () -> (T -> mpopf_model(filename, [1, .9, .8, .95, 1]; T = T, backend = backend, form = symbol))),
                        ("curve, func",  () -> (T -> mpopf_model(filename, [1, .9, .8, .95, 1], example_func; T = T, backend = backend, form = symbol))),
                        ("pregen",       () -> (T -> mpopf_model(filename, Pd_pregen, Qd_pregen; T = T, backend = backend, form = symbol))),
                        ("pregen, func", () -> (T -> mpopf_model(filename, Pd_pregen, Qd_pregen, example_func; T = T, backend = backend, form = symbol))),
                    ]
                    for (label, mk) in variants
                        build = mk()
                        m32, _, _ = build(Float32)
                        m64, _, _ = build(Float64)

                        @testset "MP w storage, $(case), $(backend), $(label), $(form_str)" begin
                            test_callbacks(m32, m64, backend)
                        end
                    end
                end
            end

            # DCOPF
            for (filename, case, _) in test_cases
                @testset "$case, DCOPF, $backend" begin
                    m32, _, _ = dcopf_model(filename; T=Float32, backend = backend)
                    m64, _, _ = dcopf_model(filename; T=Float64, backend = backend)
                    test_callbacks(m32, m64, backend)
                end
            end

            # User callbacks
            for T in (Float32, Float64)
                @testset "User callback, $(T), $(backend)" begin
                    model, vars, cons = mpopf_model(
                        "../data/pglib_opf_case3_lmbd_mod.m", elec_curve;
                        user_callback = add_electrolyzers, T=T, backend=backend)
                end

                @testset "User callback, $(T), $(backend), func" begin
                    model, vars, cons = mpopf_model(
                        "../data/pglib_opf_case3_lmbd_mod.m", elec_curve, example_func;
                        user_callback = add_electrolyzers, T=T, backend=backend)
                end
            end
        end

        if RUN_GOC3
            @testset "GOC3, Float64, nothing" begin
                sc_tests("../data/C3E4N00073D1_scenario_303", nothing, Float64)
            end
        end

        # N-1 SCOPF: CPU :single vs CPU/GPU :twostage agreement on case9, plus
        # the DC formulation. Sliced like everything else above — unguarded, the
        # CPU comparison ran in all four CI jobs instead of one.
        scopf_tests(; cpu = SELECTION in ("all", "nothing"),
                      gpu = SELECTION in ("all", "cuda") && CUDA.has_cuda_gpu())
    end
end

runtests()
