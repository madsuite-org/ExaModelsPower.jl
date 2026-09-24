# ExaModelsPower.jl
ExaModelsPower.jl is an optimal power flow models using ExaModels.jl

[![CI](https://github.com/MadNLP/ExaModelsPower.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MadNLP/ExaModelsPower.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://madsuite.org/ExaModelsPower.jl/stable/) 
[![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://madsuite.org/ExaModelsPower.jl/dev/) 
[![codecov](https://codecov.io/gh/MadNLP/ExaModelsPower.jl/graph/badge.svg?token=ybOObxcXhB)](https://codecov.io/gh/MadNLP/ExaModelsPower.jl)

## Usage
### Static optimal power flow
```julia
using ExaModelsPower, MadNLP, MadNLPGPU, CUDA, ExaModels, GOC3Benchmark, JSON


model, vars, cons = ac_opf_model(
    "pglib_opf_case118_ieee.m";
    backend = CUDABackend(),
    form = Polar()
)
result = madnlp(model; tol=1e-6)


#Alternatively, solve using DC linearization
model, vars, cons = dcopf_model(
    "pglib_opf_case118_ieee.m";
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)
```

### Security-constrained optimal power flow
```julia
#This model is based on the GOC3 formulation of the SCOPF problem
#https://www.pnnl.gov/publications/grid-optimization-competition-challenge-3-problem-formulation

#The current implementation requires a UC solution to be provided, which is then parsed with
#the other input data to generate a structure of named tuples which can then interface with 
#ExaModels to generate the full model. We do not make any relaxations or decompositions for this problem

model, cons, vars, lengths, sc_data_array = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-4)

#Solution from GPU can be used to warm start a CPU solution or vice versa
model, cons, vars, lengths, sc_data_array = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    result_set = [result, vars]
)
result_cpu = ipopt(model_cpu; tol=1e-8)

#Additionally, the SC problem can be evaluated without contingencies
model, cons, vars, lengths, sc_data_array = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    backend = CUDABackend(), include_ctg = false
)
result = madnlp(model; tol=1e-4)
```

### Generic N-1 security-constrained OPF

Independent of the GOC3 formulation, `scopf_model` builds a lightweight, hard-constrained
N-1 SCOPF. Each contingency is a generator or branch (line) outage; the base case plus all
contingencies are solved together as one OPF, with bounded corrective generator redispatch
in the post-contingency scenarios. MATPOWER `rateA = 0` ("unlimited") is mapped to a large
finite rating.

`form` takes `Polar()` (default), `Rect()` or `DC()`. A line outage is a mask, and the two
families read different fields: AC zeroes the branch admittance coefficients, DC zeroes the
branch susceptance. (Zeroing the reactance instead would divide by zero on the
zero-resistance branches real cases contain — 3 of 9 in `case9`.) Either way the outaged
line's flow is forced to zero.

```julia
using ExaModelsPower, MadNLP, DelimitedFiles

# One outage per entry; (type = :gen, idx = g) is also supported.
idxs = vec(readdlm("data/case118.Ctgs", Int))         # 1-based branch indices
contingencies = [(type = :branch, idx = l) for l in idxs]

model, vars, cons = scopf_model("case118.m", contingencies; form = Polar())
result = madnlp(model; tol = 1e-4)

# On a GPU backend, solve the condensed KKT system with cuDSS:
using MadNLPGPU, CUDA, CUDSS
model, vars, cons = scopf_model("case118.m", contingencies; backend = CUDABackend())
result = madnlp(model;
    kkt_system    = MadNLP.SparseCondensedKKTSystem,
    linear_solver = MadNLPGPU.CUDSSSolver,
    tol = 1e-4)
```

#### Two-stage / Schur form

`scopf_twostage_model` builds the same problem — AC only — on a `TwoStageExaCore` (base case =
first stage, each contingency = a scenario) so it can be solved with MadNLP's `SchurComplementCondensedKKTSystem`,
which factorizes the per-contingency blocks in parallel. It returns an extra `post_solve_info`
NamedTuple with the Schur dimensions:

```julia
model, vars, cons, info = scopf_twostage_model("case118.m", contingencies; backend = CUDABackend())
result = madnlp(model;
    callback      = MadNLP.SparseCallback,
    kkt_system    = MadNLP.SchurComplementCondensedKKTSystem,
    linear_solver = MadNLPGPU.CUDSSSolver,
    kkt_options   = Dict{Symbol,Any}(
        :schur_ns => info.ns, :schur_nv => info.nv,
        :schur_nd => info.nd, :schur_nc => info.nc,
        :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen))
```

The two scenario-tag vectors are not optional. ExaModels interleaves design and
scenario entries rather than laying them out contiguously, and MadNLP cannot
recover that partition from the dimensions alone — so `info.var_scen` and
`info.con_scen` are what let the Schur solver find the blocks.

> **Note.** The base-case physics are first-stage *design* constraints, which
> `SchurComplementCondensedKKTSystem` only learned to carry recently — hence the
> MadNLP source pin in `test/Project.toml` and `examples/Project.toml`. It comes
> out when MadNLP tags a release containing it.

A runnable example covering both solve modes (`--mode single`/`twostage`/`compare`), all three
formulations (`--form polar`/`rect`/`dc`), and a choice of case (`--case case9`/`case118`) is in
[`examples/scopf.jl`](examples/scopf.jl). Options are ArgParse CLI flags:

```bash
julia --project=./examples examples/scopf.jl --help
julia --project=./examples examples/scopf.jl --case case9 --mode twostage --form rect
julia --project=./examples examples/scopf.jl --case case9 --mode single --form dc
julia --project=./examples examples/scopf.jl --gpu --inertia free
```

`--form dc` runs under `--mode single` only, `scopf_twostage_model` being AC-only.

### Multi-period optimal power flow
```julia
model, vars, cons = mpopf_model(
    "pglib_opf_case118_ieee.m", # static network data
    "/home/sshin/git/ExaModels_Multiperiod/data/case118_onehour_168.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/case118_onehour_168.Qd"; # dynamic load data
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)

#Alternatively, input a vector to scale baseline demand to generate a demand curve
model, vars, cons = mpopf_model(
    "pglib_opf_case118_ieee.m", # static network data
    [.64, .60, .58, .56, .56, .58, .64, .76, .87, .95, .99, 1.0, .99, 1.0, 1.0,
    .97, .96, .96, .93, .92, .92, .93, .87, .72, .64], #Demand curve
    backend = CUDABackend(),
    corrective_action_ratio = 0.3
)
result = madnlp(model; tol=1e-6)

#mpopf_model can also handle inputs with storage constraints
model, vars, cons = mpopf_model(
    "pglib_opf_case30_ieee_mod.m", # static network data with storage parameters
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Qd"; # dynamic load data
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)

#Alternatively, provide a smooth function for the charge/discharge efficiency to remove complementarity constraint
function example_func(d, srating)
    return -((s_rating/2)^d)+1
end

model, vars, cons = mpopf_model(
    "pglib_opf_case30_ieee_mod.m", # static network data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Qd"; # dynamic load data
    example_func, #Discharge/charge efficiency modeled along smooth curve
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)


#Modified datasets that can be used for testing
#https://github.com/mit-shin-group/multi-period-opf-data
```

### User extension modeling
ExaModelsPower also supports the user arbitrarily extending any prebuilt models
```julia

curve = [1, .9, .95]
# Create vector of NamedTuples elec\_data w/ device data
untimed_elec_data = [(i = 1, bus = 1, cost = -5000), (i = 2, bus = 2, cost = -2000)]
Ntime = 3; Nbus = 2
elec_data = [(;b..., t = t) for b in untimed_elec_data, t in 1:Ntime]
elec_min = zeros(size(elec_data)); elec_max = fill(50, size(elec_data)); elec_scale = Float64(10)

# User-defined model modifications go here
function add_electrolyzers(core, vars, cons)
    # Add new variable to core
    p_elec = variable(core, size(elec_data, 1), size(elec_data, 2); lvar = elec_min, uvar = elec_max)
    
    # Objectives are additive. Add secondary objective
    o2 = objective(core, e.cost*p_elec[e.i, e.t] for e in elec_data)
    
    # Add electrolyzer load to power balance for each bus
    c_elec_power_balance = constraint!(core, cons.c_active_power_balance, e.bus + Nbus*(e.t-1) => p_elec[e.i, e.t] for e in elec_data)
    
    # Ramping limit over time
    c_elec_ramp = constraint(core, p_elec[e.i, e.t] - p_elec[e.i, e.t - 1] for e in elec_data[:, 2:Ntime]; lcon = fill(-elec_scale, size(elec_data[:, 2:Ntime])), ucon = fill(elec_scale, size(elec_data[:, 2:Ntime])))
    
    # Set initial electrolyzer power to 0
    c_elec_ramp_init = constraint(core, p_elec[e.i, e.t] for e in elec_data[:, 1];)

    # Return new variables and constraints to be tracked
    return ((p_elec=p_elec,), (c_elec_ramp=c_elec_ramp, c_elec_ramp_init=c_elec_ramp_init))
end
# Generate model
model, vars, cons = mpopf_model("pglib_opf_case3_lmbd.m", curve; user_callback = add_electrolyzers) # user_callback function added after initial mpopf model is constructed
```

## Citing ExaModelsPower.jl

If you use ExaModelsPower.jl in your research, we would greatly appreciate your citing it.

```bibtex
@misc{ExaModelsPower-2025,
  title  = {{ExaModelsPower.jl: A GPU-Compatible Modeling Library for Nonlinear Power System Optimization}},
  author = {Sanjay Johnson and Dirk Lauinger and Sungho Shin and François Pacaud},
  year   = {2025},
  url    = {https://arxiv.org/abs/2510.12897}, 
}
```
