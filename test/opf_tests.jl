# On a CUDA backend, MadNLP's defaults (SparseKKTSystem + MUMPS) are CPU-only and cannot
# assemble the KKT matrix from device arrays, so solve the condensed system with cuDSS on
# the GPU. On the CPU, pin the linear solver to Umfpack: MadNLP 0.10 changed the sparse
# default to MUMPS, which drives these OPFs into restoration, whereas Umfpack (the default
# through MadNLP 0.8, and what main uses) converges cleanly.
function exasolve(model, backend; kwargs...)
    opts = backend isa CUDABackend ?
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver) :
        (; linear_solver = MadNLP.UmfpackSolver)
    return madnlp(model; opts..., kwargs...)
end

function test_case3(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)

    #Branches are encoded differently in solutions, so matches are hard coded
    vars_dict =  Dict("p" => p, "q" => q)
    for st_var in ["p", "q"]
        var = vars_dict[st_var]
        for i in 1:length(result_pm["solution"]["branch"])
            @test isapprox(Array(solution(result, var))[i], result_pm["solution"]["branch"][string(i)][string(st_var, "f")], atol = result.options.tol*100)
        end
    end
end

function test_static_case(result, result_pm, result_nlp_pm, pg, qg)
    @test result.status == result_nlp_pm.status
    @test isapprox(result.objective, result_pm["objective"], rtol = result.options.tol*100)
    for i in 1:length(result_pm["solution"]["gen"])
        @test isapprox(Array(solution(result, pg))[i], result_pm["solution"]["gen"][string(i)]["pg"], atol = result.options.tol*1000)
        @test isapprox(Array(solution(result, qg))[i], result_pm["solution"]["gen"][string(i)]["qg"], atol = result.options.tol*1000)
    end
end

function test_polar_voltage(result, result_pm, va, vm)
    for i in 1:length(result_pm["solution"]["bus"])
        @test isapprox(Array(solution(result, va))[i], result_pm["solution"]["bus"][string(i)]["va"], atol = result.options.tol*100)
        @test isapprox(Array(solution(result, vm))[i], result_pm["solution"]["bus"][string(i)]["vm"], rtol = result.options.tol*100)
    end
end

function test_rect_voltage(result, result_pm, vr, vim)
    for i in 1:length(result_pm["solution"]["bus"])
        @test isapprox(Array(solution(result, vr))[i], result_pm["solution"]["bus"][string(i)]["vr"], rtol = result.options.tol*100)
        @test isapprox(Array(solution(result, vim))[i], result_pm["solution"]["bus"][string(i)]["vi"], atol = result.options.tol*100)
    end
end

function test_case5(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)
end

function test_case14(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)
end

# Every configuration is checked by evaluating the model callbacks -- objective,
# constraints, gradient and Jacobian -- and comparing the Float32 model against the
# Float64 one.  The evaluation point is the model's own initial point rather than a
# solution, so this needs no solve, which is what makes it cheap enough to run over the
# whole cross-product.
function test_callbacks(m32, m64, backend)
    x1 = m64.meta.x0
    tol = 1e-4
    x2 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    x3 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    for x in [x1, x2, x3]
        o64 = NLPModelsJuMP.obj(m64, x)
        @test isfinite(o64)
        @test isapprox(NLPModelsJuMP.obj(m32, x), o64, rtol = tol)
        c64 = NLPModelsJuMP.cons(m64, x)
        @test all(isfinite, Array(c64))
        @test isapprox(NLPModelsJuMP.cons(m32, x), c64, rtol = tol)
        if backend != CUDABackend()
            @test isapprox(NLPModelsJuMP.grad(m32, x), NLPModelsJuMP.grad(m64, x), rtol = tol)
            @test isapprox(NLPModelsJuMP.jac(m32, x), NLPModelsJuMP.jac(m64, x), rtol = tol)
        end
    end
end

function test_mp_case(result, true_sol)
    @test result.status == MadNLP.SOLVE_SUCCEEDED || result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
    @test isapprox(result.objective, true_sol, rtol = result.options.tol*100)
end

# GOC3 is a smoke test of the parser and the model constructor.  Solving it cost 32 of the
# 47 min this section took in CI, for one iteration of a 140544-variable problem, so build
# the model and evaluate its callbacks instead of solving.
#
# The model at its starting point x0 and at x = x0 + 0.01 sin.(1:nvar), as the hand-written
# JSON parser this package read GOC3 with before PowerIO built it: the two models agree
# term by term, to rounding.
const GOC3_SCENARIO_303 = (nvar = 140544, ncon = 152244,
                           obj_x0 = 0.0, abs_cons_x0 = 1.4351293014527867e9,
                           obj_x = 2.2082842501408573e-6, abs_cons_x = 1.435131605526981e9)

function sc_tests(filename, backend, T)
    uc_filename = filename*"_solution.json"
    filename = filename*".json"
    model, cons, vars, lengths, sc_data_array = ExaModelsPower.goc3_model(filename, uc_filename; backend=backend, T=T)
    x0 = model.meta.x0
    @test isfinite(NLPModelsJuMP.obj(model, x0))
    @test all(isfinite, Array(NLPModelsJuMP.cons(model, x0)))
    golden = GOC3_SCENARIO_303
    @test (model.meta.nvar, model.meta.ncon) == (golden.nvar, golden.ncon)
    @test NLPModelsJuMP.obj(model, x0) == golden.obj_x0
    @test sum(abs, Array(NLPModelsJuMP.cons(model, x0))) ≈ golden.abs_cons_x0 rtol = 1e-12
    x = copyto!(similar(x0), Array(x0) .+ 0.01 .* sin.(1:length(x0)))
    @test NLPModelsJuMP.obj(model, x) ≈ golden.obj_x rtol = 1e-9
    @test sum(abs, Array(NLPModelsJuMP.cons(model, x))) ≈ golden.abs_cons_x rtol = 1e-12
    # Every array the model is built from can move to a GPU as it is.
    for (name, v) in pairs(sc_data_array)
        v isa AbstractArray && @test isbitstype(eltype(v))
    end
    sc_uid_invariance_tests(filename, uc_filename, model, vars, lengths, sc_data_array, backend, T)
end

# The model may not depend on how a document spells its uids or orders its tables: every
# index is a position found by enumeration. (a) Rename every uid, with digits running
# against document order: the model is the same model, and the solution writer writes
# each row's values under its new name. (b) Reverse every table, which puts consumers
# before producers and the last bus first: the same model, its rows reordered.
function sc_uid_invariance_tests(case_path, uc_path, m0, vars0, lengths0, sc0, backend, T)
    case, uc = JSON.parsefile(case_path), JSON.parsefile(uc_path)
    uids = String[String(item["uid"]) for section in values(case["network"]) if section isa Vector
                  for item in section if item isa Dict && haskey(item, "uid")]
    n = length(uids)
    mapping = Dict(uid => "renamed $(n - i) ($uid)" for (i, uid) in enumerate(uids))
    rename(x::Dict) = Dict(k => rename(v) for (k, v) in x)
    rename(x::Vector) = [rename(v) for v in x]
    rename(x::String) = get(mapping, x, x)
    rename(x) = x
    dir = mktempdir()
    write_json(name, data) = (path = joinpath(dir, name); open(io -> JSON.print(io, data), path, "w"); path)
    renamed_case = write_json("renamed.json", rename(case))
    renamed_uc = write_json("renamed_solution.json", rename(uc))

    x0 = m0.meta.x0
    m1, _, vars1, lengths1, sc1 = ExaModelsPower.goc3_model(renamed_case, renamed_uc; backend = backend, T = T)
    @test (m1.meta.nvar, m1.meta.ncon) == (m0.meta.nvar, m0.meta.ncon)
    @test NLPModelsJuMP.obj(m1, x0) == NLPModelsJuMP.obj(m0, x0)
    @test Array(NLPModelsJuMP.cons(m1, x0)) == Array(NLPModelsJuMP.cons(m0, x0))

    # Write one variable vector against both documents.
    x = collect(x0)
    result = ExaModels.SolverCore.GenericExecutionStats{eltype(x),typeof(x),typeof(x),Any}(solution = x)
    written, written_renamed = joinpath(dir, "written.json"), joinpath(dir, "written_renamed.json")
    ExaModelsPower.save_go3_solution(uc_path, written, result, vars0, lengths0, sc0.uids)
    ExaModelsPower.save_go3_solution(renamed_uc, written_renamed, result, vars1, lengths1, sc1.uids)
    w0 = JSON.parsefile(written)["time_series_output"]
    w1 = JSON.parsefile(written_renamed)["time_series_output"]
    @test Set(keys(w0)) == Set(keys(w1))
    for (section, rows) in w0
        by_uid = Dict(String(r["uid"]) => r for r in w1[section])
        @test length(by_uid) == length(rows)
        @test all(row -> by_uid[rename(String(row["uid"]))] == rename(row), rows)
    end

    reverse_tables!(d) = foreach(v -> v isa Vector && reverse!(v), values(d))
    reversed, reversed_uc = deepcopy(case), deepcopy(uc)
    foreach(reverse_tables!, (reversed["network"], reversed["time_series_input"], reversed["reliability"],
                              reversed_uc["time_series_output"]))
    m2, _, _, _, _ = ExaModelsPower.goc3_model(write_json("reversed.json", reversed),
                                               write_json("reversed_solution.json", reversed_uc);
                                               backend = backend, T = T)
    @test (m2.meta.nvar, m2.meta.ncon) == (m0.meta.nvar, m0.meta.ncon)
    @test NLPModelsJuMP.obj(m2, m2.meta.x0) ≈ NLPModelsJuMP.obj(m0, x0)
    @test sort(Array(NLPModelsJuMP.cons(m2, m2.meta.x0))) ≈ sort(Array(NLPModelsJuMP.cons(m0, x0)))
end

function test_dcopf_case(result, result_pm, pg, pf)
    @test result.status == MadNLP.SOLVE_SUCCEEDED || result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
    @test isapprox(result.objective, result_pm["objective"], rtol = result.options.tol*100)
    for i in 1:length(result_pm["solution"]["gen"])
        @test isapprox(Array(solution(result, pg))[i], result_pm["solution"]["gen"][string(i)]["pg"], atol = result.options.tol*1000)
    end
    for i in 1:length(result_pm["solution"]["branch"])
        @test isapprox(Array(solution(result, pf))[i], result_pm["solution"]["branch"][string(i)]["pf"], atol = result.options.tol*1000)
    end
end
