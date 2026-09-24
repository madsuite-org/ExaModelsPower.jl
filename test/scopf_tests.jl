# Minimal N-1 SCOPF regression test.
#
# Mirrors examples/scopf.jl on the small case9 network, over three contingency
# lists (see `scopf_contingency_sets`): the 2 single-line contingencies of
# data/case9.Ctgs, a generator outage, and a list mixing the two. For each it
# solves the SAME N-1 SCOPF three ways and checks they all agree on the
# objective AND on the full base-case and per-scenario generator dispatch (the
# two formulations are mathematically identical, so a disagreement is a real
# bug, not a tolerance artifact):
#
# EVERY solve here is CONDENSED, and that is load-bearing rather than incidental.
# The Schur path is inherently condensed (RelaxEquality with bound_relax_factor =
# tol), so a monolithic solve left on the DEFAULT sparse KKT optimizes the
# hard-equality problem instead and the two disagree by the relaxation, not by a
# bug. Measured on case9/K=2/CPU/Float64: matching the relaxation gives objective
# gap 0.0 and dispatch gap 2.12e-13, where leaving `:single` hard gives 13.43 and
# 2.12e-3 -- and that 13.43 is ~2*tol of equality slack, so it scales WITH tol
# (1.4e-3 at tol=1e-8) and cannot be tuned away by tightening.
#
#   * CPU :single    — scopf_model            (monolithic ExaModel, condensed KKT + CHOLMOD)
#   * CPU :twostage  — scopf_twostage_model    (Schur complement KKT + MUMPS)
#   * GPU :twostage  — scopf_twostage_model    (Schur complement KKT + cuDSS), skipped
#                                               when no CUDA GPU is present
#
# CPU :single is the trusted reference. GPU :single is deliberately NOT tested: the
# condensed-KKT GPU path has been observed to report success at a point that does not
# satisfy power balance, so it cannot serve as the thing everything else is checked
# against.

# Build the SchurComplementCondensedKKTSystem kkt_options from the model's post_solve_info
# tags (MadNLP can't auto-detect ExaModels' interleaved design/scenario tag names).
function scopf_schur_kkt_options(info)
    return Dict{Symbol,Any}(
        :schur_ns => info.ns, :schur_nv => info.nv, :schur_nd => info.nd, :schur_nc => info.nc,
        :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen,
    )
end

# Solve the monolithic scopf_model. `vars.pg` is ngen × (K+1): column 1 is the base
# case, columns 2..K+1 are the K contingencies. Returns (result, pg).
function solve_scopf_single(case, contingencies, backend)
    model, vars, _ = scopf_model(case, contingencies; backend = backend)
    opts = backend isa CUDABackend ?
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver) :
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver)
    result = madnlp(model; tol = 1.0e-4, print_level = MadNLP.ERROR, opts...)
    return result, Array(solution(result, vars.pg))
end

# Solve the two-stage scopf_twostage_model via the Schur complement KKT system (MUMPS
# on CPU, cuDSS on GPU). `vars.pg0` is the base dispatch (ngen); `vars.pgk` the per-
# scenario dispatch (ngen × K). Returns (result, pg0, pgk).
function solve_scopf_twostage(case, contingencies, backend; inertia = MadNLP.InertiaBased)
    model, vars, _, info = scopf_twostage_model(case, contingencies; backend = backend)
    lin = backend isa CUDABackend ? MadNLPGPU.CUDSSSolver : MadNLP.MumpsSolver
    result = madnlp(model;
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SchurComplementCondensedKKTSystem,
        linear_solver = lin,
        kkt_options = scopf_schur_kkt_options(info),
        inertia_correction_method = inertia,
        tol = 1.0e-4, print_level = MadNLP.ERROR,
    )
    return result, Array(solution(result, vars.pg0)), Array(solution(result, vars.pgk))
end

converged(r) = r.status == MadNLP.SOLVE_SUCCEEDED || r.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

# Assert a two-stage solve agrees with the reference single-case dispatch. Both
# layouts are generator-fast / scenario-slow, so compare flattened (robust to whether
# `solution` hands back a matrix or a flat vector for the per-scenario variable).
function test_scopf_agrees(r, pg0, pgk, r_ref, pg0_ref, pgk_ref)
    @test converged(r)
    @test isapprox(r.objective, r_ref.objective, rtol = 1.0e-3)
    @test isapprox(vec(pg0), vec(pg0_ref), atol = 1.0e-3)   # base-case dispatch
    @test isapprox(vec(pgk), vec(pgk_ref), atol = 1.0e-3)   # per-scenario dispatch
end

# What a generator outage means for the reference dispatch `pg` (ngen × (K+1)),
# checked on the solution because agreement between the two models would not
# catch a trip that both of them ignore. In the scenario of a tripped unit that
# unit produces nothing, and every other unit sits at its base dispatch plus a
# corrective adjustment in `[0, ratio * pmax]` — so the base dispatch of the
# tripped unit is what the survivors' headroom can cover, which is the whole
# point of the model. The `atol` is the solver's bound relaxation: a condensed
# solve leaves a zero bound at ~tol rather than at zero.
function test_scopf_gen_outages(case, contingencies, pg; ratio = 0.05, atol = 1.0e-3)
    pmax = ExaModelsPower.parse_ac_power_data(case, Float64).pmax
    for (k, ct) in enumerate(contingencies)
        ct.type == :gen || continue
        g = ct.idx
        c = k + 1                                   # scenario 1 is the base case
        live = setdiff(axes(pg, 1), g)
        Δ = pg[live, c] .- pg[live, 1]
        @test abs(pg[g, c]) < atol                                     # the unit tripped
        @test pg[g, 1] > 10 * atol                                     # it was running before
        @test all(Δ .>= -atol)                                         # survivors only ramp up
        @test all(Δ .<= ratio .* pmax[live] .+ atol)                   # within their cap
        @test sum(Δ) >= pg[g, 1] - 10 * atol                           # and cover the lost output
    end
end

# The DC formulation of the same N-1 problem. `scopf_twostage_model` is AC-only,
# so there is no second DC solve to check against — this checks the MECHANISM
# the DC outage rests on instead.
#
# In AC an outage zeroes the admittance coefficients `c1..c8`; in DC those play
# no part, and the outage is a zero written into the susceptance `bs` that
# `_scopf_narrow` carries on each branch row. If that zero failed to reach the
# flow equation the branch would keep conducting, and the model would still
# build, still converge, and still report a plausible cost — so the flow itself
# is what is asserted. A generator outage needs no DC-specific mask (it is a
# bound on `pg` in every form), but it is checked here too, for the same reason:
# the coupling row it drops is exact to see in the constraint matrix.
#
# Asserted on the FORMULATION rather than on a solution. The DC model is linear, so
# "the outaged line carries no flow" is a property of the constraint matrix: it holds
# exactly when `pf[l, c]` is pinned to a constant by the equalities alone and that
# constant is zero. Checking it there is what this package is responsible for, it is
# exact rather than tolerance-bound, and it holds at every feasible point rather than
# at one — whether a particular SCOPF instance is solvable is a separate question
# from whether it is formulated correctly.
#
# Armed on both sides: nothing may pin the same line in the BASE case (a formulation
# that pinned every flow would otherwise pass), and nothing may pin a different line
# in the SAME scenario (a dead scenario would otherwise pass). The arming doubles as
# a check on the index arithmetic, which is the one place this reaches into ExaModels'
# variable layout: a mis-aimed index would have to land on a variable pinned to
# exactly zero in precisely the outaged (line, scenario) pairs and unpinned in both
# controls.

# The equality block of a linear model, as `A * x == b`.
function linear_equality_block(model)
    n, m = model.meta.nvar, model.meta.ncon
    A = Matrix(NLPModels.jac(model, zeros(n)))
    c0 = NLPModels.cons(model, zeros(n))
    x = collect(range(0.1, 1.3; length = n))          # everything below reads the
    @test maximum(abs, NLPModels.cons(model, x) .- (A * x .+ c0)) < 1.0e-10   # model off `A`
    eq = findall(i -> model.meta.lcon[i] == model.meta.ucon[i], 1:m)
    return A[eq, :], model.meta.ucon[eq] .- c0[eq]
end

# Do the equalities alone pin `x[j]` to a constant? Returns how far the unit vector
# `e_j` sits from the row space of `A` — zero exactly when pinned — and the constant
# it is pinned to.
function pinned_value(A, b, n, j)
    e = zeros(n)
    e[j] = 1.0
    y = A' \ e
    return maximum(abs, A' * y .- e), LinearAlgebra.dot(y, b)
end

function test_scopf_dc(case, contingencies)
    model, vars, _ = scopf_model(case, contingencies; form = DC())
    nbranch, nscen = vars.pf.size
    @test nscen == length(contingencies) + 1

    A, b = linear_equality_block(model)
    @test LinearAlgebra.rank(A) == size(A, 1)   # so the least squares in `pinned_value` is meaningful
    n = model.meta.nvar
    pf(l, c) = vars.pf.offset + (c - 1) * nbranch + l
    ngen = vars.pg.size[1]
    pg(g, c) = vars.pg.offset + (c - 1) * ngen + g
    lvar, uvar = Array(model.meta.lvar), Array(model.meta.uvar)
    fixed_at_zero(j) = lvar[j] == 0 && uvar[j] == 0
    # Is there an equality row tying `x[i]` to `x[j]`? The corrective coupling
    # `pg[g, c] - pg[g, 1] - extra[g, c]` is the only one that ties a unit's
    # dispatch across scenarios.
    coupled(i, j) = any(r -> !iszero(A[r, i]) && !iszero(A[r, j]), axes(A, 1))

    for (k, ct) in enumerate(contingencies)
        c = k + 1                                 # scenario 1 is the base case
        if ct.type == :branch
            l = ct.idx
            resid, value = pinned_value(A, b, n, pf(l, c))
            @test resid < 1.0e-10                                        # the outage happened
            @test abs(value) < 1.0e-10                                   # and it zeroed the flow
            @test first(pinned_value(A, b, n, pf(l, 1))) > 1.0e-3        # not so in the base case
            @test first(pinned_value(A, b, n, pf(l == 1 ? 2 : 1, c))) > 1.0e-3   # nor for a live line
        else
            # A generator outage is a bound, not a mask: the tripped unit's
            # dispatch is fixed to zero in its own scenario, and its coupling
            # row is dropped — kept, it would drag the base dispatch to zero
            # with it. Armed as above: the unit is free in the base case, and a
            # live unit in the same scenario is neither fixed nor uncoupled.
            g = ct.idx
            live = g == 1 ? 2 : 1
            @test fixed_at_zero(pg(g, c))                                # the unit tripped
            @test !coupled(pg(g, c), pg(g, 1))                           # and left the base free
            @test !fixed_at_zero(pg(g, 1))                               # not so in the base case
            @test !fixed_at_zero(pg(live, c))                            # nor for a live unit,
            @test coupled(pg(live, c), pg(live, 1))                      # which is still coupled
        end
    end
end

# The contingency lists every test below runs over, as `name => list`.
#
# A generator outage on case9 survives only because the base case keeps the
# tripped unit low enough for the other two to cover it within their 5%
# corrective cap — so one generator per list: case9 has three units, and no
# base dispatch leaves two of them that low while meeting the load (the solver
# reports two generator outages infeasible). Gen 3 is the one the mixed list
# trips, placed between the two lines so a generator scenario is neither first
# nor last and the base-case column shift is exercised on both sides of it.
function scopf_contingency_sets()
    # Each line of case9.Ctgs is a 1-based branch index to outage (just like the example).
    ctg_idxs = parse.(Int, filter(!isempty, strip.(readlines(joinpath(@__DIR__, "..", "data", "case9.Ctgs")))))
    branches = [(type = :branch, idx = l) for l in ctg_idxs]
    return [
        "branch" => branches,
        "gen" => [(type = :gen, idx = 1)],
        "gen+branch" => [branches[1], (type = :gen, idx = 3), branches[2:end]...],
    ]
end

function scopf_tests(; cpu = true, gpu = CUDA.has_cuda_gpu())
    (cpu || gpu) || return nothing

    case = joinpath(@__DIR__, "..", "data", "case9.m")
    for (name, contingencies) in scopf_contingency_sets()
        scopf_tests(case, name, contingencies; cpu = cpu, gpu = gpu)
    end
    return nothing
end

function scopf_tests(case, name, contingencies; cpu, gpu)
    K = length(contingencies)

    @testset "SCOPF case9 N-1 $name (K=$K)" begin
        # CPU :single is the reference solution. The GPU comparison needs it too,
        # so it is not gated on `cpu`.
        r_single, pg_single = solve_scopf_single(case, contingencies, nothing)
        @test converged(r_single)
        pg0_ref = pg_single[:, 1]
        pgk_ref = pg_single[:, 2:end]

        if any(ct -> ct.type == :gen, contingencies)
            @testset "generator outages hold in the reference dispatch" begin
                test_scopf_gen_outages(case, contingencies, pg_single)
            end
        end

        if cpu
            @testset "CPU two-stage matches single" begin
                r, pg0, pgk = solve_scopf_twostage(case, contingencies, nothing)
                test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
            end

            @testset "DC outage removes the outaged element" begin
                test_scopf_dc(case, contingencies)
            end
        end

        if gpu
            @testset "GPU two-stage matches single" begin
                r, pg0, pgk = solve_scopf_twostage(case, contingencies, CUDABackend())
                test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
            end
        end
    end
    return nothing
end
