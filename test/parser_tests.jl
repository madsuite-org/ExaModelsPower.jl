# Reading a case: PowerIO parses it, and `parse_ac_power_data` keeps what is in
# service, in file order, as the rows the models read.

import PowerIO

# `case` with the rows of `table` (as word vectors) replaced by `f(rows)`.
function mp_map(f, case, table)
    out = String[]
    rows = Vector{Vector{String}}()
    intab = false
    for line in readlines(case)
        if !intab
            push!(out, line)
            intab = occursin(Regex("^\\s*mpc\\.$table\\s*=\\s*\\["), line)
        elseif occursin(']', split(line, '%')[1])
            append!(out, ["\t" * join(r, '\t') * ";" for r in f(rows)])
            push!(out, line)
            intab = false
        else
            w = split(strip(replace(split(line, '%')[1], ';' => ' ')))
            isempty(w) || push!(rows, String.(w))
        end
    end
    path = tempname() * ".m"
    write(path, join(out, '\n'))
    return path
end

renumber(rows, cols) = [(r = copy(r); for c in cols; r[c] = string(parse(Int, r[c]) + 1000); end; r) for r in rows]
withcol(rows, pred, col, v) = [pred(r) ? (r = copy(r); r[col] = v; r) : r for r in rows]
mp_append(case, text) = (path = tempname() * ".m"; write(path, read(case, String) * "\n" * text); path)

# Rows as flat vectors of numbers, compared to rounding: a case written to another
# format and read back passes through MW and per unit again.
flat(x::Number) = [Float64(x)]
flat(x::Union{Tuple,NamedTuple}) = reduce(vcat, map(flat, Tuple(x)); init = Float64[])
rows_close(a, b) = length(a) == length(b) &&
    all(isapprox(flat(r), flat(s); rtol = 1e-12, atol = 1e-12) for (r, s) in zip(a, b))

const STORAGE_ROWS = """
mpc.storage = [
	1009	 0.0	 0.0	 1.00	 200.0	 100.0	 72.0	 0.90	 0.85	 1000.0	 -1000.0	 1000.0	 0.1	 0.01	 5.0	 2.0	 1;
	1003	 0.0	 0.0	 1.00	 200.0	 100.0	 72.0	 0.90	 0.85	 1000.0	 -1000.0	 1000.0	 0.1	 0.01	 0.0	 0.0	 0;
];
"""

function test_parser(case14)
    @testset "parse: inference" begin
        # The compiled library calls these with a path; every field must infer
        # concretely or `juliac --trim=safe` refuses the call.
        @test isconcretetype(Base.infer_return_type(ExaModelsPower.opf_args,
                                                    (String, Type{Float64}, Nothing, typeof((;)))))
        @test isconcretetype(Base.infer_return_type(ExaModelsPower.opf_args,
                                                    (PowerIO.PioModule{PowerIO.BalancedNetwork}, Type{Float32}, Nothing, typeof((;)))))
        @test isconcretetype(Base.infer_return_type(mpopf_args_default, (String,)))
        @test isconcretetype(Base.infer_return_type(mpopf_args,
                                                    (String, Vector{Float64}, Int, Type{Float32}, Nothing, Float64)))
        d = ExaModelsPower.parse_ac_power_data(case14, Float32)
        for t in (:bus, :gen, :branch, :arc, :storage)     # case14 has no storage
            @test isbitstype(eltype(getfield(d, t)))
        end
        @test eltype(d.storage) == ExaModelsPower.StorageRow{Float32}
    end

    @testset "parse: in-service rows, file order" begin
        # Bus numbers offset by 1000 and the bus rows reversed: positions follow
        # the file, references follow the numbers. Bus 8 isolated, with its
        # generator and its one branch (7-8) out of service; storage at bus
        # 1009 in service, at bus 1003 out of service.
        isolated = mp_map(rows -> withcol(rows, r -> r[1] == "1008", 2, "4"),
                          mp_map(rows -> reverse(renumber(rows, (1,))), case14, "bus"), "bus")
        edited = mp_map(rows -> withcol(renumber(rows, (1,)), r -> r[1] == "1008", 8, "0"), isolated, "gen")
        edited = mp_map(rows -> withcol(renumber(rows, (1, 2)), r -> (r[1], r[2]) == ("1007", "1008"), 11, "0"),
                        edited, "branch")
        edited = mp_append(edited, STORAGE_ROWS)
        d = ExaModelsPower.parse_ac_power_data(edited)
        ref = ExaModelsPower.parse_ac_power_data(case14)

        @test length(d.bus) == 13 && length(d.gen) == 4 && length(d.branch) == 19
        @test [b.bus_i for b in d.bus] == [1014:-1:1009; 1007:-1:1001]
        @test [b.i for b in d.bus] == 1:13
        @test [g.i for g in d.gen] == 1:4
        @test [d.bus[g.bus].bus_i for g in d.gen] == [1001, 1002, 1003, 1006]
        m = length(d.branch)
        @test [br.i for br in d.branch] == 1:m
        @test all(br.f_idx == br.i && br.t_idx == br.i + m for br in d.branch)
        @test [(d.bus[br.f_bus].bus_i, d.bus[br.t_bus].bus_i) for br in d.branch] ==
              [(ref.bus[br.f_bus].bus_i + 1000, ref.bus[br.t_bus].bus_i + 1000)
               for br in ref.branch if (ref.bus[br.f_bus].bus_i, ref.bus[br.t_bus].bus_i) != (7, 8)]
        @test [a.bus for a in d.arc] == [[br.f_bus for br in d.branch]; [br.t_bus for br in d.branch]]
        @test d.ref_buses == [findfirst(b -> b.bus_i == 1001, d.bus)]

        @test length(d.storage) == 1
        s = only(d.storage)
        @test s.i == 1 && d.bus[s.storage_bus].bus_i == 1009
        @test s.p_loss ≈ 0.05 && s.q_loss ≈ 0.02          # per unit, like every other power
        @test length(d.emax) == 1

        # The edited case builds and solves, storage included.
        mp, _, _ = mpopf_model(edited, [1.0, 0.95]; form = Polar())
        @test madnlp(mp; print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "parse: refused" begin
        isolated = mp_map(rows -> withcol(rows, r -> r[1] == "8", 2, "4"), case14, "bus")
        @test_throws ArgumentError ExaModelsPower.parse_ac_power_data(isolated)    # gen and branch at bus 8
        @test_throws ArgumentError ExaModelsPower.parse_ac_power_data(
            mp_map(rows -> withcol(rows, r -> r[1] == "8", 8, "0"), isolated, "gen"))  # branch 7-8 still in service
        # MATPOWER's gencost is a matrix: the polynomial rows are padded to the
        # piecewise row's width.
        piecewise = mp_map(rows -> [[["1", "0", "0", "2", "0", "0", "100", "2000"]];
                                    [[r; "0"] for r in rows[2:end]]], case14, "gencost")
        @test_throws ArgumentError ExaModelsPower.parse_ac_power_data(piecewise)
        @test_throws ArgumentError ExaModelsPower.parse_ac_power_data("no_such_case.m")
        @test_throws ArgumentError ExaModelsPower._case_path("/no/such/pglib_opf_case14_ieee.m")
    end

    @testset "parse: a rating of 0 is the implied thermal limit" begin
        # MATPOWER reads rateA = 0 as no limit; PowerModels' calc_thermal_limits!
        # states it as the limit the voltage and angle bounds imply.
        unrated = mp_map(rows -> withcol(rows, r -> r[1] in ("1", "2"), 6, "0"), case14, "branch")
        d = ExaModelsPower.parse_ac_power_data(unrated)
        pm = parse_pm(unrated)
        for (l, br) in enumerate(d.branch)
            @test br.rate_a ≈ pm["branch"][string(l)]["rate_a"]
            @test d.arc[l].rate_a == d.arc[l + length(d.branch)].rate_a == br.rate_a
        end
        @test all(isfinite(br.rate_a) && br.rate_a > 0 for br in d.branch)
    end

    @testset "parse: a network PowerIO already parsed" begin
        @test ExaModelsPower.opf_args(PowerIO.parse(PowerIO.BalancedNetwork, case14)) == ExaModelsPower.opf_args(case14)
        m = PowerIO.parse(PowerIO.PioModule{PowerIO.BalancedNetwork}, case14)
        @test ExaModelsPower.opf_args(m) == ExaModelsPower.opf_args(case14)
        # An update applied in PowerIO reaches the model.
        PowerIO.apply_updates!(m, [PowerIO.set_load_active_power(PowerIO.ComponentId("load", "bus-2"),
                                                                 PowerIO.ActivePower(megawatts = 30.0))])
        @test ExaModelsPower.parse_ac_power_data(m).bus[2].pd ≈ 0.30
    end

    @testset "parse: other formats" begin
        m = PowerIO.parse(PowerIO.PioModule{PowerIO.BalancedNetwork}, case14)
        json = tempname() * ".json"
        PowerIO.emit(m, "powermodels-json", json)
        dj, ref = ExaModelsPower.parse_ac_power_data(json), ExaModelsPower.parse_ac_power_data(case14)
        for t in (:bus, :gen, :branch, :arc, :storage)
            @test rows_close(getfield(dj, t), getfield(ref, t))
        end
        obj(c) = madnlp(ac_opf_model(c)[1]; print_level = MadNLP.ERROR).objective
        @test obj(json) ≈ obj(case14) rtol = 1e-8

        # PSS/E RAW carries no costs and orders transformers after lines, but the
        # same network reaches the model.
        raw = tempname() * ".raw"
        PowerIO.emit(m, "psse", raw)
        d = ExaModelsPower.parse_ac_power_data(raw)
        @test (length(d.bus), length(d.gen), length(d.branch)) == (length(ref.bus), length(ref.gen), length(ref.branch))
        @test rows_close([(b.pd, b.qd, b.bs) for b in d.bus], [(b.pd, b.qd, b.bs) for b in ref.bus])
        @test rows_close(sort([(br.br_r, br.br_x) for br in d.branch]), sort([(br.br_r, br.br_x) for br in ref.branch]))
        @test madnlp(ac_opf_model(raw)[1]; print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "parse: case lookup" begin
        @test ExaModelsPower._case_path(case14) == case14
        p = ExaModelsPower._case_path("pglib_opf_case5_pjm.m")
        @test isfile(p) && endswith(p, joinpath("pglib-opf-23.07", "pglib_opf_case5_pjm.m"))
        @test isfile(ExaModelsPower._case_path(joinpath("api", "pglib_opf_case5_pjm__api.m")))
    end

    @testset "matpower output: rows follow the parse" begin
        # Bus 8 isolated, with its generator and branch out of service: the model
        # leaves them out and the writer gives their rows zeros.
        isolated = mp_map(rows -> withcol(rows, r -> r[1] == "8", 2, "4"), case14, "bus")
        isolated = mp_map(rows -> withcol(rows, r -> r[1] == "8", 8, "0"), isolated, "gen")
        isolated = mp_map(rows -> withcol(rows, r -> (r[1], r[2]) == ("7", "8"), 11, "0"), isolated, "branch")
        out, _, _, r = mp_solved(isolated, Polar())
        @test r.status == MadNLP.SOLVE_SUCCEEDED
        bus = mp_table(out, "bus")
        @test length(bus) == 14
        @test all(iszero, bus[8][14:17]) && bus[8][8] == 1.0     # the input's VM, untouched

        # A solution of another case, and a case that is not MATPOWER, are refused.
        m, v, c = ac_opf_model(case14)
        x, y, z = m.meta.x0, zeros(m.meta.ncon), zeros(m.meta.nvar)
        @test_throws DimensionMismatch write_matpower_solution(tempname() * ".m", "../data/pglib_opf_case5_pjm.m",
                                                               v, c, x, y, z, z)
        json = tempname() * ".json"
        PowerIO.emit(PowerIO.parse(PowerIO.PioModule{PowerIO.BalancedNetwork}, case14), "powermodels-json", json)
        @test_throws ArgumentError write_matpower_solution(tempname() * ".m", json, v, c, x, y, z, z)
    end

    @testset "multi-period loads" begin
        # A curve scales the loads; bus 9's shunt is a fixed admittance and keeps its value.
        d, = mpopf_args(case14, [1.0, 0.5])
        b1, b2 = d.busarray[9, 1].b, d.busarray[9, 2].b
        @test b2.pd ≈ b1.pd / 2 && b2.qd ≈ b1.qd / 2
        @test b1.bs == b2.bs > 0 && b1.gs == b2.gs
        @test_throws DimensionMismatch mpopf_args(case14, [1.0, 0.5]; N = 3)

        pd, qd = tempname(), tempname()
        write(pd, join((join(fill("10.0", 2), ' ') for _ in 1:14), '\n'))
        write(qd, join((join(fill("5.0", 2), ' ') for _ in 1:14), '\n'))
        m, _, _ = mpopf_model(case14, pd, qd)
        @test m.meta.nvar > 0
        short = tempname()
        write(short, join((join(fill("10.0", 2), ' ') for _ in 1:13), '\n'))
        @test_throws DimensionMismatch mpopf_model(case14, short, qd)
        @test_throws DimensionMismatch mpopf_model(case14, pd, qd; N = 3)
    end
end

# PGLib cases by name, solved against the PowerModels/Ipopt reference.
function test_pglib_reference(names)
    ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    for name in names
        @testset "PGLib $name, AC polar == PowerModels" begin
            r = exasolve(ac_opf_model(name)[1], nothing; print_level = MadNLP.ERROR)
            ref = solve_opf(ExaModelsPower._case_path(name), ACPPowerModel, ipopt)
            @test r.status == MadNLP.SOLVE_SUCCEEDED
            @test isapprox(r.objective, ref["objective"]; rtol = 1e-6)
        end
    end
end
