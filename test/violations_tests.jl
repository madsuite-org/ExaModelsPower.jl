# opf_violations: zero (to the solver's tolerance) at a solution, and each group measures
# its own blocks, which a perturbation of one block shows.
function test_violations(case)
    for (name, form) in (("polar", Polar()), ("rect", Rect()), ("dc", DC()))
        @testset "violations, $name" begin
            m, vars, cons = opf_model(case; form)
            r = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-8)
            v = opf_violations(m, r.solution)
            @test keys(v) == (:power_balance, :power_flow, :branch_flow, :angle_difference, :voltage, :generation)
            @test all(x -> 0 <= x < 1e-6, v)

            # Generation 0.5 p.u. above its upper bound: only `generation` moves.
            x = copy(r.solution)
            i = vars.pg.offset + 1
            x[i] = m.meta.uvar[i] + 0.5
            w = opf_violations(m, x)
            @test w.generation ≈ 0.5
            # Power balance at that generator's bus moves by the change in its output.
            @test w.power_balance ≈ x[i] - r.solution[i] atol = 1e-6

            # The writer writes the same numbers.
            out = tempname()
            write_violations(out, case, m, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U)
            rows = split.(readlines(out)[2:end], ',')
            @test first.(rows) == string.(collect(keys(v)))
            @test parse.(Float64, last.(rows)) == collect(v)

            # The reports a compiled library carries: the instance, and a solution's cost and
            # violations by group.
            ri = opf_report_instance(case, m)
            @test startswith(ri, basename(case) * ": ")
            @test occursin(string(m.meta.nvar, " variables, ", m.meta.ncon, " constraints"), ri)
            @test occursin(" buses, ", ri) && occursin(" generators, ", ri) && occursin(" branches; ", ri)
            rs = opf_report_solution(case, m, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U;
                                     objective = r.objective, status = "solved")
            @test startswith(rs, "generation cost ")
            @test length(split(rs, '\n')) == 1 + length(v)          # one line per group
            @test occursin("power balance", rs) && occursin("generation", rs)
        end
    end
end
