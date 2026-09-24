# Standalone N-1 SCOPF example.
#
# Solves a security-constrained OPF over the single-line (N-1) contingencies
# listed in data/<case>.Ctgs, with bounded corrective generator redispatch. Two
# formulations of the same problem are available via `mode`:
#
#   :single    scopf_model          — one monolithic ExaModel; condensed sparse KKT
#                                      (CHOLMOD on CPU, cuDSS on GPU).
#   :twostage  scopf_twostage_model — TwoStageExaCore (base case = first stage,
#                                      each contingency = a scenario); solved with
#                                      MadNLP's SchurComplementCondensedKKTSystem.
#   :compare   solve BOTH and assert they reach the same objective and base-case
#              dispatch (the two formulations are mathematically identical).
#
# Run from the examples env (options have defaults; pass --help to list them):
#   julia --project=./examples examples/scopf.jl
#   julia --project=./examples examples/scopf.jl --case case9 --mode twostage --form rect
#   julia --project=./examples examples/scopf.jl --case case9 --mode single --form dc
#   julia --project=./examples examples/scopf.jl --gpu --inertia free
#
# `--form dc` is :single only — scopf_twostage_model is AC-only.
#
# GPU two-stage (Schur) converges reliably to the true optimum on case9 at the default
# tol=1e-4, the Schur assembly being deterministic:
#   julia --project=./examples examples/scopf.jl --case case9 --mode twostage --gpu --inertia based

using ExaModelsPower
using ExaModels: solution
using MadNLP
using MadNLPGPU
using CUDA
using CUDSS   # activates MadNLPGPUCUDAExt so MadNLPGPU.CUDSSSolver is the real cuDSS solver
using DelimitedFiles
using ArgParse

const DATADIR = joinpath(pkgdir(ExaModelsPower), "data")

# ---- Options ----------------------------------------------------------------
function parse_options(args)
    s = ArgParseSettings(description = "Standalone N-1 SCOPF example (scopf_model / scopf_twostage_model).")
    @add_arg_table! s begin
        "--case"
            help = "case name; needs data/<case>.m + data/<case>.Ctgs (e.g. case9, case118)"
            arg_type = String
            default = "case118"
        "--mode"
            help = ":single (scopf_model), :twostage (scopf_twostage_model), or :compare (both, asserts agreement)"
            arg_type = Symbol
            default = :compare
            range_tester = m -> m in (:single, :twostage, :compare)
        "--form"
            help = "formulation: polar, rect (voltage coordinates) or dc; dc needs --mode single"
            arg_type = Symbol
            default = :polar
            range_tester = f -> f in (:polar, :rect, :dc)
        "--gpu"
            help = "solve on the GPU (CUDABackend); default is CPU"
            action = :store_true
        "--inertia"
            help = "MadNLP inertia_correction_method: auto, based, or free (inertia-free)"
            arg_type = Symbol
            default = :auto
            range_tester = i -> i in (:auto, :based, :free)
        "--max-iter"
            help = "MadNLP max_iter (cap iterations; useful for studying non-convergent configs)"
            arg_type = Int
            dest_name = "max_iter"
            default = 3000
        "--cudss-ir"
            help = "cuDSS iterative-refinement steps for the GPU Schur path (scenario + complement solvers); 0 = Schur-path default (5)"
            arg_type = Int
            dest_name = "cudss_ir"
            default = 0
        "--matching"
            help = "cuDSS matching (pivot-maximizing permutation) on the GPU solvers. The factorization/solve are correct, but cuDSS 0.8 reports inertia (0,0) whenever matching is on, so --inertia based/auto spiral into RESTORATION_FAILED; use --inertia free, or wait for a diag-based inertia fallback in MadNLPGPU. Off by default until that lands."
            arg_type = Bool
            default = false
        "--tol"
            help = "MadNLP tol. GPU two-stage has a ~5e-3 Schur backward-error floor; tol below ~3e-3 mostly stalls there. Use ~5e-3."
            arg_type = Float64
            default = 1.0e-4
        "--richardson"
            help = "MadNLP richardson_max_iter (outer iterative-refinement cap). 0 = library default (10). GPU two-stage benefits from ~50."
            arg_type = Int
            default = 0
        "--retry"
            help = "retry a non-converged solve up to N times. GPU two-stage convergence is a per-run roulette (~60%); ~6 → >99%."
            arg_type = Int
            default = 1
    end
    return parse_args(args, s; as_symbols = true)
end

# Build the two-stage Schur `kkt_options` from the model's two-stage tags. On GPU,
# configure cuDSS iterative refinement on both solvers, and (if requested) matching on
# the first-stage Schur-complement solver ONLY — cuDSS matching is NOT SUPPORTED in
# uniform-batch mode (analysis fails with CUDSS_STATUS_NOT_SUPPORTED), and the
# per-scenario blocks are one ubatch solver.
# On the complement solver matching factorizes/solves correctly but cuDSS 0.8 reports
# inertia (0,0) with matching on (see --matching help), so inertia-based IPM modes fail
# until MadNLPGPU recovers the inertia from the factor diagonal. An explicit
# `schur_*_opt_linear_solver` struct REPLACES MadNLPGPU's Schur-path defaults
# (cudss_ir = 5, cudss_ir_tol = 0.0), so those must be re-applied here — in
# particular ir_tol must stay 0.0 to keep cuDSS 0.8's IR_FAILED gate disarmed.
function schur_kkt_options(info, backend, cudss_ir, matching)
    kkt = Dict{Symbol, Any}(
        :schur_ns => info.ns, :schur_nv => info.nv, :schur_nd => info.nd, :schur_nc => info.nc,
        :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen,
    )
    if backend !== nothing && (cudss_ir > 0 || matching)
        for (key, allow_matching) in ((:schur_scenario_opt_linear_solver, false),
                                      (:schur_opt_linear_solver, true))
            o = MadNLP.default_options(MadNLPGPU.CUDSSSolver)
            o.cudss_ir = cudss_ir > 0 ? cudss_ir : 5
            o.cudss_ir_tol = 0.0
            o.cudss_matching = matching && allow_matching
            kkt[key] = o
        end
    end
    return kkt
end

const INERTIA = Dict(
    :auto  => MadNLP.InertiaAuto,
    :based => MadNLP.InertiaBased,
    :free  => MadNLP.InertiaFree,
)

# The command line spells a formulation as a word; the models dispatch on a type.
const FORM = Dict(:polar => Polar(), :rect => Rect(), :dc => DC())

opts     = parse_options(ARGS)
casename = opts[:case]
mode     = opts[:mode]
form     = FORM[opts[:form]]

# `scopf_twostage_model` is AC-only, so :twostage and :compare have nothing to run
# under DC. Refuse up front rather than fail inside the second solve.
opts[:form] === :dc && mode !== :single &&
    error("--form dc supports --mode single only (scopf_twostage_model is AC-only), got --mode $(mode)")
backend  = opts[:gpu] ? CUDABackend() : nothing
inertia  = INERTIA[opts[:inertia]]   # MadNLP inertia_correction_method type
max_iter = opts[:max_iter]
cudss_ir = opts[:cudss_ir]
matching = opts[:matching]
tol        = opts[:tol]
richardson = opts[:richardson]
retry      = opts[:retry]

# Pass richardson_max_iter only when overridden (0 = library default of 10).
rich_kw() = richardson > 0 ? (; richardson_max_iter = richardson) : (;)

# Every mode solves a condensed KKT system so they all optimize the SAME relaxation:
# the two-stage Schur path is inherently condensed (RelaxEquality with
# bound_relax_factor = tol), and a non-condensed `:single` (hard equalities) would
# reach the true optimum, ~2·tol of equality slack away (13.43 on case9 at tol=1e-4).
# The condensed matrix is SPD: CHOLMOD on CPU, cuDSS on GPU.
condensed_opts() = backend === nothing ?
    (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver) :
    (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver,
       cudss_matching = matching)

# GPU two-stage convergence is a per-run roulette (the GPU Schur apply is not backward
# stable on the ill-conditioned KKT) AND can occasionally converge (status SUCCESS) to a
# ~3% suboptimal local point — so status alone is NOT a correctness check. Retry up to
# `retry` times, accepting only a converged solve whose objective matches the reference
# `ref` (if given) within `reltol`. Returns the first accepted result, else the last.
function solve_with_retry(buildsolve; ref = nothing, reltol = 1.0e-3)
    local r
    for attempt in 1:retry
        r = buildsolve()
        ok = r.status == MadNLP.SOLVE_SUCCEEDED &&
             (ref === nothing || abs(r.objective - ref) <= reltol * max(abs(ref), 1.0))
        ok && return r
        attempt < retry && @info "solve not accepted; retrying" attempt status = r.status objective = r.objective
    end
    return r
end

# Each line of <case>.Ctgs is a 1-based branch index to outage.
ctg_idxs = vec(readdlm(joinpath(DATADIR, "$casename.Ctgs"), Int))
contingencies = [(type = :branch, idx = l) for l in ctg_idxs]
case = joinpath(DATADIR, "$casename.m")

@info "Building SCOPF" casename mode form backend inertia n_contingencies = length(contingencies)

if mode == :single
    model, vars, cons = scopf_model(case, contingencies; form = form, backend = backend)
    @info "Model size" n_var = model.meta.nvar n_con = model.meta.ncon

    result = madnlp(model; tol = tol, print_level = MadNLP.INFO,
                    inertia_correction_method = inertia, max_iter = max_iter, condensed_opts()...)

    println()
    @info "SCOPF result" status = result.status objective = result.objective iterations = result.iter
    pg = Array(solution(result, vars.pg))
    println("\nBase-case active dispatch pg[:, 1] (p.u.):")
    println(round.(pg[:, 1]; digits = 3))

elseif mode == :twostage
    model, vars, cons, info = scopf_twostage_model(case, contingencies; form = form, backend = backend)
    @info "Model size" n_var = model.meta.nvar n_con = model.meta.ncon schur = info

    # Schur KKT: both the per-scenario blocks AND the first-stage Schur complement
    # are sparse — the complement fills only the coupled base-dispatch design vars.
    # `linear_solver` here factorizes that sparse complement: MUMPS on CPU, cuDSS on
    # GPU (both symmetric-indefinite, report inertia). The scenario dimensions and
    # per-variable/per-constraint scenario tags come from the model's two-stage tags;
    # the tags let the Schur solver partition the interleaved layout and fold the
    # `nc_design` base-case design constraints into its first-stage block.
    lin = backend === nothing ? MadNLP.MumpsSolver : MadNLPGPU.CUDSSSolver
    kkt_opts = schur_kkt_options(info, backend, cudss_ir, matching)

    # With the deterministic GPU Schur assembly the two-stage solve converges reliably at
    # tol=1e-4 (matching CPU); --retry defaults to 1 (a single solve) and is kept only as a
    # safety net.
    result = solve_with_retry() do
        madnlp(model;
            callback = MadNLP.SparseCallback,
            kkt_system = MadNLP.SchurComplementCondensedKKTSystem,
            linear_solver = lin,
            kkt_options = kkt_opts,
            inertia_correction_method = inertia,
            max_iter = max_iter,
            tol = tol, print_level = MadNLP.INFO,
            rich_kw()...,
        )
    end
    println()
    @info "SCOPF (two-stage) result" status = result.status objective = result.objective iterations = result.iter
    pg0 = Array(solution(result, vars.pg0))
    println("\nBase-case active dispatch pg0 (p.u.):")
    println(round.(pg0; digits = 3))

elseif mode == :compare
    # Solve the SAME N-1 SCOPF both ways and check they agree. Both use the same
    # condensed relaxation (see condensed_opts) and the same tol, so the objectives
    # must match tightly.
    m1, v1, _ = scopf_model(case, contingencies; form = form, backend = backend)
    r1 = madnlp(m1; tol = tol, print_level = MadNLP.ERROR,
                inertia_correction_method = inertia, max_iter = max_iter, condensed_opts()...)
    pg_single = Array(solution(r1, v1.pg))[:, 1]

    m2, v2, _, info = scopf_twostage_model(case, contingencies; form = form, backend = backend)
    # Sparse first-stage Schur complement solver: MUMPS on CPU, cuDSS on GPU. The
    # per-scenario sparse blocks use cuDSS/MUMPS internally.
    lin = backend === nothing ? MadNLP.MumpsSolver : MadNLPGPU.CUDSSSolver
    r2 = madnlp(m2;
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SchurComplementCondensedKKTSystem,
        linear_solver = lin,
        kkt_options = schur_kkt_options(info, backend, cudss_ir, matching),
        inertia_correction_method = inertia,
        max_iter = max_iter,
        tol = tol, print_level = MadNLP.ERROR,
        rich_kw()...,
    )
    pg0 = Array(solution(r2, v2.pg0))

    obj_gap = abs(r1.objective - r2.objective)
    pg_gap = maximum(abs, pg_single .- pg0)
    println()
    @info "Equivalence" single_obj = r1.objective twostage_obj = r2.objective objective_gap = obj_gap base_dispatch_gap = pg_gap
    @assert r1.status == MadNLP.SOLVE_SUCCEEDED && r2.status == MadNLP.SOLVE_SUCCEEDED "a solve did not converge"
    @assert obj_gap < 1.0e-3 * max(1.0, abs(r1.objective)) "objective mismatch: $obj_gap"
    @assert pg_gap < 1.0e-3 "base dispatch mismatch: $pg_gap"
    println("\n:single and :twostage agree (objective gap $(round(obj_gap; sigdigits = 3)), base dispatch gap $(round(pg_gap; sigdigits = 3))).")

else
    error("unknown mode $mode; use :single, :twostage, or :compare")
end
