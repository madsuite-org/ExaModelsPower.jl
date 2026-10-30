# start = :case: the starting point is the case file's voltages and dispatch, so a
# model built from a solved case starts at that solution, and solves in fewer iterations.
function test_case_start(case)
    for (name, form, blocks) in (("polar", Polar(), (:va, :vm, :pg, :qg)),
                                 ("rect", Rect(), (:vr, :vim, :pg, :qg)),
                                 ("dc", DC(), (:va, :pg)))
        @testset "case start, $name" begin
            m, v, c = opf_model(case; form)
            r = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-8)
            sol = tempname() * ".m"
            write_matpower_solution(sol, case, v, c, r.solution, r.multipliers, r.multipliers_L,
                                    r.multipliers_U; form, objective = r.objective, status = "solved")

            ws, wv, _ = opf_model(sol; form, start = :case)
            blk(h, a) = a[(h.offset + 1):(h.offset + ExaModels.total(h.size))]
            for b in blocks
                @test blk(wv[b], ws.meta.x0) ≈ blk(v[b], r.solution) rtol = 1e-10 atol = 1e-12
            end
            # The other blocks (the flows) keep the default start.
            flat, _, _ = opf_model(sol; form)
            nb = sum(ExaModels.total(wv[b].size) for b in blocks)
            @test count(ws.meta.x0 .!= flat.meta.x0) <= nb

            # case_start places the same values into a model's own start.
            @test case_start(sol, flat) == ws.meta.x0
            @test case_start(sol, m) == ws.meta.x0
            out = tempname()
            write_case_start(out, sol, m, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U)
            @test parse.(Float64, readlines(out)) == ws.meta.x0

            # Started at the solution, the AC solve takes fewer iterations (the DC one, a
            # quadratic program, no more).
            rw = madnlp(ws; print_level = MadNLP.ERROR, tol = 1e-8)
            @test rw.status == MadNLP.SOLVE_SUCCEEDED
            @test rw.objective ≈ r.objective rtol = 1e-6
            @test form isa DC ? rw.iter <= r.iter : rw.iter < r.iter
        end
    end
    @test_throws ArgumentError opf_model(case; start = :flat)
    # A case with other buses is refused rather than misplaced.
    m5, _, _ = opf_model("../data/pglib_opf_case5_pjm.m")
    @test_throws DimensionMismatch case_start(case, m5)
end
