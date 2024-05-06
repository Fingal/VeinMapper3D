## Requirements

Certain parts of the project were created using Cython. The repositiory comes with precompiled binares for Anaconda3-2024.02. It is required to recompile all cython files in `cythin_files` folder and copy binares to the main folder.

## Running simulation

to run specific simulation one has to invoke coresponding function, namely:

- `run_experiment.run_fibonacci()` for Fibonacci phylotaxy 
- `run_experiment.run_bijugate()` for bijugate phylotaxy 
- `run_experiment.run_lucas()` for Lucas phylotaxy 

## Parameter names

Folowing table coresponds names of parameters in article to their coresponding names in the code

| Name in article  | Name in code |
| ------------- | ------------- |
| $G$ radial growth rate of the apex  | global variable `grow_coeffs` in `simulation_initialization.py`  |
| $d_r$ minimal distance between the apex center $c$ and iV endpoint (µm)  |  global variable `meristem_distances` in `simulation_initialization.py`  |
| plastochron length (h)  |  global variable  `surface_frequency` in `simulation_initialization.py`  |
| iV stage when the connection to DR5 maxima is established  | cotrolled by atribute `extra_offset` in `SurfacePoints`  |
| iV stage when repulsion is introduced  | cotrolled by atribute `matured_age` in `GlobalSimulation`  |
| $V_r$ iV extension rate (µm/h)  | global variable `growth_rate` in `simulation_initialization.py`  |
| $b_r$ age of iV bifurcating on right side (h)  | global variable `n5_ages` in `simulation_initialization.py`  |
| $b_l$ age of iV bifurcating on right side (h)  | global variable `n8_ages` in `simulation_initialization.py`  |
| $d_b$ distance between the bifurcation point and iV merging point (µm)  | Cotrolled by atribute `YOUNG_DISTANCE` in `GlobalSimulation`  |
| $s_n$ repulsion strength   | Controlled by atribute `coeffs["aainhibition_coefa"]` in `GlobalSimulation` |
| $r_n$ repulsion range (µm)	 | Controlled by atribute `coeffs["neg_range"]` in `GlobalSimulation` |
| $s_p$ attraction strength	 | Controlled by atribute `coeffs["attraction_coef"]` in `GlobalSimulation` |
| $r_p$ attraction range (µm) | Controlled by atribute `coeffs["pos_range"]` in `GlobalSimulation` |
| $s_s$ attraction to the apex center  | Controlled by atribute `coeffs["straight_coef"]` in `GlobalSimulation` |
| $s_A$ inertia | Controlled by atribute `coeffs["inertia"]` in `GlobalSimulation` |
| $p_p$ peak position | Controlled by atribute `coeffs["peak_coef"]` in `GlobalSimulation` |
| $a_m$ age modifier | Controlled by atribute `coeffs["age_coef"]` in `GlobalSimulation` |
| $a_min$ age cutoff | Controlled by atribute `coeffs["age_cut_off_coef"]` in `GlobalSimulation` |
| interconnection attraction threshold (µm) | Controlled by atribute `young_attraction_distance` in `GlobalSimulation` |
| interconnection attraction strength | Controlled by atribute `young_attraction_strength` in `GlobalSimulation` |
| interconnection threshold (µm) | Controlled by atribute `connection_distance` in `GlobalSimulation` |
| primordium connection distance | Controlled by atribute `primordium_connection_distance` in `GlobalSimulation` |
| primordium divergence angle | Controlled by atribute `angle_offest` in `SurfacePoints` |