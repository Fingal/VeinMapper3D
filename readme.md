## Requirements

Certain parts of the project were created using Cython. The repository comes with precompiled binaries for Anaconda3-2024.02. It is required to recompile all cython files in `cython_files` folder and copy binaries to the main folder.

## Running simulation

to run specific simulation one has to invoke corresponding function, namely:

- `run_experiment.run_fibonacci()` for Fibonacci phyllotaxis 
- `run_experiment.run_bijugate()` for bijugate phyllotaxis 
- `run_experiment.run_lucas()` for Lucas phyllotaxis 

## Parameter names

Following table corresponds names of parameters in article to their corresponding names in the code

| Name in article  | Name in code |
| ------------- | ------------- |
| $G$ radial growth rate of the apex  | global variable `grow_coeffs` in `simulation_initialization.py`  |
| $d_r$ minimal distance between the apex center $c$ and iV endpoint (µm)  |  global variable `meristem_distances` in `simulation_initialization.py`  |
| plastochron length (h)  |  global variable  `surface_frequency` in `simulation_initialization.py`  |
| iV stage when the connection to DR5 maxima is established  | controlled by attribute `extra_offset` in `SurfacePoints`  |
| iV stage when repulsion is introduced  | controlled by attribute `matured_age` in `GlobalSimulation`  |
| $V_r$ iV extension rate (µm/h)  | global variable `growth_rate` in `simulation_initialization.py`  |
| $b_r$ age of iV bifurcating on right side (h)  | global variable `n5_ages` in `simulation_initialization.py`  |
| $b_l$ age of iV bifurcating on right side (h)  | global variable `n8_ages` in `simulation_initialization.py`  |
| $d_b$ distance between the bifurcation point and iV merging point (µm)  | Controlled by attribute `YOUNG_DISTANCE` in `GlobalSimulation`  |
| $s_n$ repulsion strength   | Controlled by attribute `coeffs["aainhibition_coefa"]` in `GlobalSimulation` |
| $r_n$ repulsion range (µm)	 | Controlled by attribute `coeffs["neg_range"]` in `GlobalSimulation` |
| $s_p$ attraction strength	 | Controlled by attribute `coeffs["attraction_coef"]` in `GlobalSimulation` |
| $r_p$ attraction range (µm) | Controlled by attribute `coeffs["pos_range"]` in `GlobalSimulation` |
| $s_s$ attraction to the apex center  | Controlled by attribute `coeffs["straight_coef"]` in `GlobalSimulation` |
| $s_A$ inertia | Controlled by attribute `coeffs["inertia"]` in `GlobalSimulation` |
| $p_p$ peak position | Controlled by attribute `coeffs["peak_coef"]` in `GlobalSimulation` |
| $a_m$ age modifier | Controlled by attribute `coeffs["age_coef"]` in `GlobalSimulation` |
| $a_min$ age cutoff | Controlled by attribute `coeffs["age_cut_off_coef"]` in `GlobalSimulation` |
| interconnection attraction threshold (µm) | Controlled by attribute `young_attraction_distance` in `GlobalSimulation` |
| interconnection attraction strength | Controlled by attribute `young_attraction_strength` in `GlobalSimulation` |
| interconnection threshold (µm) | Controlled by attribute `connection_distance` in `GlobalSimulation` |
| primordium connection distance | Controlled by attribute `primordium_connection_distance` in `GlobalSimulation` |
| primordium divergence angle | Controlled by attribute `angle_offest` in `SurfacePoints` |
| angular threshold | Controlled by attribute `angle_error` in `SurfacePoints` |

## Comparison to random walk

Comparison between a skeleton and random walk (Fig. 4A) is done using `DataContext.sample_random_distance` method in `batch_scripts/distance_calculation.py`

## PCA Analysis

Images 4F-H were done using `PCA_analisys/notebook.ipynb`
