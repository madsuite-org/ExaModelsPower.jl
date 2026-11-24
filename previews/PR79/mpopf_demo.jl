# # [Multi-period OPF](@id mpopf_demo)

# The multi-period (MP) OPF can also be modeled in ExaModelsPower with a number of user-specified adjustments, including storage. 

# Some of the models in this portion of the tutorial involve using external files. While we provide the necessary code to access these additional files, you may view them __[here](https://github.com/mit-shin-group/multi-period-opf-data)__ as well.
# We will start with the simplest way to model the MPOPF, which also does not require the user to have any data already downloaded. Instead, the user specifies a demand curve for the system. The demand curve is a vector of ratios from 0 to 1 which indicate the scaling of demand compared to the demand indicated by the static OPF file. In this model of the MPOPF, every consuming bus has the same scaling in power demand for each point in time. A corrective action ratio, which limits the ramp rate of generators, can also be inputted. It is set to 0.1 as a default. The adjustable coordinate system and backend that were present for the static OPF are also available for all MPOPF models.
using ExaModelsPower, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels
model, vars, cons = mpopf_model(
    "pglib_opf_case118_ieee.m", # static network data
    [.64, .60, .58, .56, .56, .58, .64, .76, .87, .95, .99, 1.0, .99, 1.0, 1.0,
    .97, .96, .96, .93, .92, .92, .93, .87, .72, .64], #Demand curve
    backend = CUDABackend(),
    corrective_action_ratio = 0.3
)
result = madnlp(model; tol=1e-6, kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver)

# If the user would like to provide more complex demand profiles, they can provide their own input files. The number of rows in each input file should match the number of buses in the static OPF datafile, and the number of columns will dictate the number of time periods in the MP model.
# First, download the example load profile datafiles.
using Downloads

#Define URLs for Pd and Qd data files (raw or raw GitHub links)
pd_url = "https://raw.githubusercontent.com/mit-shin-group/multi-period-opf-data/refs/heads/main/halfhour_30.Pd"
qd_url = "https://raw.githubusercontent.com/mit-shin-group/multi-period-opf-data/refs/heads/main/halfhour_30.Qd"

#Define local paths to temporarily store the files
pd_file = "halfhour_30.Pd"
qd_file = "halfhour_30.Qd"

#Download the files if they don't already exist
if !isfile(pd_file)
    Downloads.download(pd_url, pd_file)
end

if !isfile(qd_file)
    Downloads.download(qd_url, qd_file);
end

# Next, build the MPOPF model, providing the dynamic load data instead of a demand curve as input.
#Run your model
model, vars, cons = mpopf_model(
    "pglib_opf_case30_ieee.m",  # static network data (assumed local or already handled)
    pd_file,                    # dynamic load data (Pd)
    qd_file;                    # dynamic load data (Qd)
    backend = CUDABackend()
)

#Solve
result = madnlp(model; tol=1e-6, kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver)

## MPOPF with storage
# The MPOPF model can also be constructed with storage considerations. We model storage using the model proposed in __[Geth, Coffrin, Fobes 2020](https://arxiv.org/pdf/2004.14768)__. This requires inputting a modified datafile containing storage parameters. When modeling MPOPF with storage, all of the aforementioned tuneable parameters are still available. We also allow the user to specify whether or not to model the charging/discharging complementarity constraint. This is set to false by default to avoid potential numerical error.

#First, we download the modified datafile with storage parameters included.
#Define URL for the main datafile
stor_url = "https://raw.githubusercontent.com/mit-shin-group/multi-period-opf-data/refs/heads/main/pglib_opf_case30_ieee_mod.m"

#Define local path to temporarily store the file
stor_file = "pglib_opf_case30_ieee_mod.m"

#Download the file if it doesn't already exist
if !isfile(stor_file)
    Downloads.download(stor_url, stor_file);
end

# Generate the model with your modified datafile. If the datafile contains storage parameters, ExaModelsPower will automatically recognize it and include the additional necessary constraints. 
model, vars, cons = mpopf_model(
    stor_file, # static network data with storage parameters
    pd_file,                    # dynamic load data (Pd)
    qd_file;                    # dynamic load data (Qd)
    backend = CUDABackend(),
    storage_complementarity_constraint = false
)
result = madnlp(model; tol=1e-6, kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver)
result.objective

# ExaModelsPower also provides a secondary option to avoid dealing with complementarity constraints. The user can specify a function that computesloss in battery level as a smooth function of discharge rate and the storage devices thermal rating parameter. We provide an arbitrary example function to demonstrate the modeling capability.

function example_function(d, srating)
    return d + .2/srating*d^2
end;

# ExaModelsPower will automatically adjust the necessary constraints if one of the inputs provided is a function.
model, vars, cons = mpopf_model(
    stor_file, # static network data with storage parameters
    pd_file,                    # dynamic load data (Pd)
    qd_file,                    # dynamic load data (Qd)
    example_function;           # discharge function
    backend = CUDABackend(),
    storage_complementarity_constraint = false
)
result = madnlp(model; tol=1e-6, kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver)
result.objective

# Despite the example discharge function being generated somewhat arbitrarily, the resultant objective values remain quite close for both the smooth and piecewise charge/discharge functions.

