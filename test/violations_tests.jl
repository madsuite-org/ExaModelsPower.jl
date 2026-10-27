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
            # Power balance at that generator's bus moves by the same 0.5.
            @test w.power_balance ≈ 0.5 atol = 1e-6

            # The writer writes the same numbers.
            out = tempname()
            write_violations(out, case, m, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U)
            rows = split.(readlines(out)[2:end], ',')
            @test first.(rows) == string.(collect(keys(v)))
            @test parse.(Float64, last.(rows)) == collect(v)
        end
    end
end
