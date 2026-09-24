import NLPModels

# The recipe, the eagerly-built core, and the model a caller actually gets must
# all be the same model. They are three spellings of one body, and the point of
# the split is that they cannot drift — so that is checked here rather than
# assumed.

function test_recipe_equivalence(filename, form; T = Float64)
    # recipe ∘ args — which is what `ac_opf_model` is
    m1, vars, _ = ac_opf_model(filename; form = form, T = T)
    # the same body built eagerly, data in hand — what `ExaModelsC` compiles
    # The eager model, built directly: same body, data already in the core.
    data, = ExaModelsPower.opf_args(filename; T = T)
    core, _, _ = ExaModelsPower.build_opf(ExaCore(T), form, data,
                                          ExaModelsPower.dummy_extension, T)
    m2 = ExaModel(core)

    @test m1.meta.nvar == m2.meta.nvar
    @test m1.meta.ncon == m2.meta.ncon
    @test m1.meta.nnzj == m2.meta.nnzj
    @test m1.meta.nnzh == m2.meta.nnzh
    @test Array(m1.meta.x0) == Array(m2.meta.x0)
    @test Array(m1.meta.lvar) == Array(m2.meta.lvar)
    @test Array(m1.meta.uvar) == Array(m2.meta.uvar)
    @test Array(m1.meta.lcon) == Array(m2.meta.lcon)
    @test Array(m1.meta.ucon) == Array(m2.meta.ucon)

    # Evaluated away from x0, so a difference in the starting point and one in
    # the algebra cannot hide behind one another.
    x = T[1 + T(0.01) * sin(i) for i = 1:m1.meta.nvar]
    y = T[T(0.01) * cos(i) for i = 1:m1.meta.ncon]
    @test NLPModels.obj(m1, x) == NLPModels.obj(m2, x)
    @test NLPModels.grad(m1, x) == NLPModels.grad(m2, x)
    @test NLPModels.cons(m1, x) == NLPModels.cons(m2, x)
    @test NLPModels.jac_coord(m1, x) == NLPModels.jac_coord(m2, x)
    @test NLPModels.hess_coord(m1, x, y; obj_weight = T(0.7)) ==
          NLPModels.hess_coord(m2, x, y; obj_weight = T(0.7))
end

# The same guarantee for the security-constrained model. It is the one whose
# recipe carries a symbolic `Nbus` into arithmetic -- the balance appends index
# row `a.bus + Nbus * (k - 1)` -- so recipe and eager agreeing here is a
# stronger statement than it is for the static model, where nothing is offset.
function test_scopf_recipe_equivalence(filename, form;
                                       T = Float64,
                                       contingencies = ExaModelsPower.SCOPF_DEFAULT_CONTINGENCIES)
    K = length(contingencies)
    m1, _, _ = scopf_model(filename, contingencies; form = form, T = T)

    data, = ExaModelsPower.scopf_args(filename, contingencies; K = K, T = T)
    core, _, _ = ExaModelsPower.build_scopf_body(ExaCore(T), form, data, K,
                                                 length(data.bus),
                                                 ExaModelsPower.dummy_extension, T)
    m2 = ExaModel(core)

    @test m1.meta.nvar == m2.meta.nvar
    @test m1.meta.ncon == m2.meta.ncon
    @test m1.meta.nnzj == m2.meta.nnzj
    @test m1.meta.nnzh == m2.meta.nnzh
    @test Array(m1.meta.x0) == Array(m2.meta.x0)
    @test Array(m1.meta.lvar) == Array(m2.meta.lvar)
    @test Array(m1.meta.uvar) == Array(m2.meta.uvar)
    @test Array(m1.meta.lcon) == Array(m2.meta.lcon)
    @test Array(m1.meta.ucon) == Array(m2.meta.ucon)

    x = T[1 + T(0.01) * sin(i) for i = 1:m1.meta.nvar]
    y = T[T(0.01) * cos(i) for i = 1:m1.meta.ncon]
    @test NLPModels.obj(m1, x) == NLPModels.obj(m2, x)
    @test NLPModels.grad(m1, x) == NLPModels.grad(m2, x)
    @test NLPModels.cons(m1, x) == NLPModels.cons(m2, x)
    @test NLPModels.jac_coord(m1, x) == NLPModels.jac_coord(m2, x)
    @test NLPModels.hess_coord(m1, x, y; obj_weight = T(0.7)) ==
          NLPModels.hess_coord(m2, x, y; obj_weight = T(0.7))
end

# REGRESSION. The handles `ac_opf_model` returns come out of the RECIPE, where
# an offset is still an `ArgNode` expression rather than a number. Unless they
# are resolved against the arguments the model was built from, `solution`
# indexes the solution vector with a symbolic range and throws
#
#     ArgumentError: invalid index: (((0 + (_length(length(arg.bus)) * 1)) + …
#
# a long way from the cause. This shipped and the suite caught it.
function test_solution_handles(filename, form)
    m, vars, _ = ac_opf_model(filename; form = form)
    result = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-6)
    @test result.status == MadNLP.SOLVE_SUCCEEDED
    # Each call throws outright on an un-instantiated handle; the lengths then
    # say the blocks are the ones they claim to be, and together they must
    # account for every variable in the model.
    total = 0
    for k in keys(vars)
        s = Array(solution(result, vars[k]))
        @test !isempty(s)
        total += length(s)
    end
    @test total == m.meta.nvar
end

# Compiling is minutes of juliac, so this runs ONCE for the whole suite rather
# than once per case and formulation: one call, then a few properties of what
# comes back. `obj`/`cons` are checked away from x0, so a library that returned
# constants -- or that ignored the case it was handed -- would fail rather than
# agree trivially.
function test_aot()
    # A directory of our own rather than the `@emp` default. A `@name` path
    # resolves against CNLPMODELS_PATH, which nothing in this repo sets, so the
    # default would fail this test on the environment rather than on the models.
    r = compile_all(ExaModelsPower; path = mkpath(joinpath(mktempdir(), "emp")))
    @test isfile(r.libpath)
    @test Set(Symbol.(r.prefixes)) ==
          Set((:acp, :acr, :dcp, :mpacp, :mpacr, :mpdcp, :scacp, :scacr, :scdcp))

    case = "pglib_opf_case14_ieee.m"
    m = CNLPModel(r.libpath, :acp, case)
    ref, _, _ = ac_opf_model(case)
    @test (m.meta.nvar, m.meta.ncon) == (ref.meta.nvar, ref.meta.ncon)
    @test collect(m.meta.x0) == collect(ref.meta.x0)
    x = [0.9 + 0.02sin(i) for i = 1:m.meta.nvar]
    @test NLPModels.obj(m, x) == NLPModels.obj(ref, x)
    @test collect(NLPModels.cons(m, x)) == collect(NLPModels.cons(ref, x))

    # The property the recipe rewrite exists for: a network size the library
    # was never compiled against.
    m3 = CNLPModel(r.libpath, :mpdcp, "pglib_opf_case3_lmbd.m")
    ref3, _, _ = mpopf_model("pglib_opf_case3_lmbd.m",
                             ExaModelsPower.MPOPF_DEFAULT_CURVE;
                             form = ExaModelsPower.DC())
    @test (m3.meta.nvar, m3.meta.ncon) == (ref3.meta.nvar, ref3.meta.ncon)

    # Same again for the security-constrained model, whose baked structure is a
    # contingency LIST rather than a horizon -- the axis most likely to be got
    # wrong when only the case changes underneath it.
    m4 = CNLPModel(r.libpath, :scacp, "pglib_opf_case3_lmbd.m")
    ref4, _, _ = scopf_model("pglib_opf_case3_lmbd.m",
                             ExaModelsPower.SCOPF_DEFAULT_CONTINGENCIES)
    @test (m4.meta.nvar, m4.meta.ncon) == (ref4.meta.nvar, ref4.meta.ncon)
    x4 = [0.9 + 0.02sin(i) for i = 1:m4.meta.nvar]
    @test NLPModels.obj(m4, x4) == NLPModels.obj(ref4, x4)
    @test collect(NLPModels.cons(m4, x4)) == collect(NLPModels.cons(ref4, x4))
end
