# write_matpower_solution: the solved case must read back as a valid MATPOWER case whose
# result columns agree with the solution and with each other.

# Numeric rows of `mpc.<table>` in file order.
function mp_table(file, table)
    rows = Vector{Float64}[]
    intab = false
    for line in readlines(file)
        if !intab
            intab = occursin(Regex("^\\s*mpc\\.$table\\s*=\\s*\\["), line)
            continue
        end
        code = split(line, '%')[1]
        occursin(']', code) && break
        w = split(strip(replace(code, ';' => ' ', ',' => ' ')))
        isempty(w) || push!(rows, parse.(Float64, w))
    end
    return rows
end

# `case` with column `col` of row `row` of `table` replaced by `f(value)`.
function mp_edit(case, table, row, col, f)
    out = String[]
    intab = false
    k = 0
    for line in readlines(case)
        if !intab
            intab = occursin(Regex("^\\s*mpc\\.$table\\s*=\\s*\\["), line)
        elseif occursin(']', split(line, '%')[1])
            intab = false
        else
            w = split(strip(replace(split(line, '%')[1], ';' => ' ')))
            if !isempty(w) && (k += 1) == row
                w[col] = string(f(parse(Float64, w[col])))
                line = "\t" * join(w, '\t') * ";"
            end
        end
        push!(out, line)
    end
    path = tempname() * ".m"
    write(path, join(out, '\n'))
    return path
end

function mp_solve(case, form)
    m, vars, cons = ac_opf_model(case; form)
    r = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-10)
    return m, vars, cons, r
end

function mp_solved(case, form)
    _, vars, cons, r = mp_solve(case, form)
    out = tempname() * ".m"
    write_matpower_solution(out, case, vars, cons, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U;
                            form, objective = r.objective, status = string(r.status))
    return out, vars, cons, r
end

function test_matpower_output(case)
    # One generator and one branch out of service, so the zero rows are exercised.
    base_case = mp_edit(mp_edit(case, "gen", 5, 8, _ -> 0), "branch", 5, 11, _ -> 0)
    files = Dict{String,String}()
    for (form_str, form) in (("polar", Polar()), ("rect", Rect()), ("dc", DC()))
        @testset "matpower output, $form_str" begin
            out, vars, cons, r = mp_solved(base_case, form)
            files[form_str] = out
            @test r.status == MadNLP.SOLVE_SUCCEEDED
            @test startswith(readline(out), "%% Solved by ExaModelsPower")

            bus, gen, branch = mp_table(out, "bus"), mp_table(out, "gen"), mp_table(out, "branch")
            @test length(bus) == length(mp_table(base_case, "bus"))
            @test length(gen) == length(mp_table(base_case, "gen"))
            @test length(branch) == length(mp_table(base_case, "branch"))
            @test all(length.(bus) .== 17) && all(length.(gen) .== 25) && all(length.(branch) .== 21)

            # PowerModels reads it as a case.
            data = PowerModels.parse_file(out)
            @test length(data["bus"]) == length(bus)

            @test all(iszero, gen[5][[2, 3, 22, 23, 24, 25]])
            @test all(iszero, branch[5][14:21])

            # Power balance at every bus from the written file alone, in MW.
            mis = Dict(Int(b[1]) => -b[3] - b[5] * b[8]^2 for b in bus)
            for g in gen
                g[8] != 0 && (mis[Int(g[1])] += g[2])
            end
            for l in branch
                l[11] != 0 || continue
                mis[Int(l[1])] -= l[14]
                mis[Int(l[2])] -= l[16]
            end
            @test maximum(abs, values(mis)) < 1e-6

            # LAM_P is the marginal cost of load: d objective / d PD.
            i = findfirst(b -> b[3] > 0, bus)
            δ = 0.01
            fd = (mp_solve(mp_edit(base_case, "bus", i, 3, p -> p + δ), form)[4].objective -
                  mp_solve(mp_edit(base_case, "bus", i, 3, p -> p - δ), form)[4].objective) / 2δ
            @test isapprox(bus[i][14], fd; rtol = 1e-4)
            @test bus[i][14] > 0

            # Rewriting a solved case replaces the result columns rather than appending more.
            again = tempname() * ".m"
            write_matpower_solution(again, out, vars, cons, r.solution, r.multipliers, r.multipliers_L, r.multipliers_U; form)
            @test mp_table(again, "bus") == bus
            @test mp_table(again, "gen") == gen
            @test mp_table(again, "branch") == branch
        end
    end
    @testset "matpower output, polar == rect" begin
        bp, br = mp_table(files["polar"], "bus"), mp_table(files["rect"], "bus")
        @test isapprox(reduce(vcat, (b[8] for b in bp)), reduce(vcat, (b[8] for b in br)); atol = 1e-6)
        @test isapprox(reduce(vcat, (b[14] for b in bp)), reduce(vcat, (b[14] for b in br)); rtol = 1e-5)
    end
end
