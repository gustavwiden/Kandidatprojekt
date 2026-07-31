# A minimal PBPK model predicts pDC-dependent decoupling of litifilimab skin exposure from plasma pharmacokinetics in Systemic Lupus Erythematosus (SLE)

This repository contains the code necessary for recreating all results presented in the article. The implementation was done in Python (3.13.2) using the [Simulation Using Non-linear dynamics (SUND) toolbox](https://doi.org/10.48550/arXiv.2510.13932).

## Prerequisites
To run this repo locally, clone the project and install the dependencies from `requirements.txt`. 

**Using `uv`:**
If you use the `uv` package manager, you can install the dependencies and execute the scripts with:
```bash
uv pip install -r requirements.txt
uv run Scripts/plot_simulations.py
```

## Plot the results from the article
Run the script `plot_simulations.py` to generate the plots for Figures 1-7 and Supplementary Figures 1-6.

> **Note:** For Supplementary Figures 7-10, see the Identifiability Analysis section below.

## Reproduce the complete model development
Follow the step-by-step workflow presented below:

1. **Parameter Estimation:**
   Run the script `run_parameter_estimation.py`. You will need to run this script three separate times, manually changing the global variable `pDC_density` at the top of the script to `'1'`, `'80'`, and `'400'`. This estimates the best parameter set used in the subsequent steps.

2. **Identifiability Analysis:**
   Run the script `run_identifiability_analysis.py`. 
   * The `run_mcmc` function only needs to be executed once. 
   * The `run_profile_likelihood` function should be run for both `1_dgf` and `N_dgf` modes across all three `pDC_density` levels to generate the plots for Supplementary Figures 7-10 and the acceptable parameter sets used to plot simulation uncertainty bands in step 4.

3. **Update Parameters:**
   The scripts in steps 1 and 2 output their parameter sets into the `Results/` folder. To plot these newly generated simulation uncertainty bands and main simulation lines, you must manually move the new `.csv` and `.json` files from `Results/` into the `Parameters/` folder, overwriting the existing published parameters.

4. **Plot Simulations:**
   Run `plot_simulations.py` to generate the plots for Figures 1-7 and Supplementary Figures 1-6. 
   
> **Note:** The function `plot_skin_plasma_concentration_ratio` in the script `plot_simulations.py` requires minor manual commenting/uncommenting to switch between plotting for SLE patients (Figure 7B) and HV (Figure 7C).
---

## Structure of the repository

### Data/
This folder contains the data `.json` files needed to run the scripts.

**Digitised literature data:**
The following files are digitised clinical data from the **Phase I trial**
- `HV_PK_data.json`
- `HV_PD_data.json`
- `SLE_PK_data.json`
- `SLE_PD_data.json`

The following file is from the **Phase II trial**
- `SLE_CLE_PK_validation_data.json`

**Generated data:**
Additional `.json` files are created automatically when running `plot_simulations.py`:
- `HV_vs_SLE_plasma_PD_response_data.json`
- `HV_vs_SLE_plasma_PK_response_data.json`
- `SLE_IV_dose_skin_PD_response_data.json`
- `SLE_skin_SC_clinical_menu_loading_1.json`
- `SLE_skin_SC_clinical_menu_loading_80.json`
- `SLE_skin_SC_clinical_menu_loading_400.json`
- `SLE_skin_SC_min_maintenance_1.json`
- `SLE_skin_SC_min_maintenance_80.json`
- `SLE_skin_SC_min_maintenance_400.json`

> **Note:** Running `plot_simulations.py` will automatically regenerate and overwrite these initially provided datasets to mimic the actual workflow which was used during model development. In this way, the generated datasets always reflect the most recent simulations.

### Models/
This folder contains the model `.txt` files needed to run the scripts.

**Models for SLE patients:**
There are three models representing SLE patients. These reflect different degrees of lesional pDC infiltration and thus BDCA2 target burden:
- `SLE_model_1.txt` (1 pDC/mm² in skin)
- `SLE_model_80.txt` (80 pDCs/mm² in skin)
- `SLE_model_400.txt` (400 pDCs/mm² in skin)

**Models for healthy volunteers:**
There are two models representing healthy volunteers. These reflect the normal and high-end pDC plasma concentration:
- `HV_model.txt` (5100 pDCs/L in plasma)
- `HV_model_high.txt`* (10200 pDCs/L in plasma)

> **Note:** The high-end model is only used in `plot_simulations.py` to generate the biodistribution plot for Figure 7C.

### Parameters/
This folder contains the parameter `.json` and `.csv` files needed to run the scripts.

**Simulation uncertainty:**
All acceptable parameter sets found during the Profile Likelihood (PL) analysis (`mode='N_dgf'`) have been gathered into a single file for each pDC density. These files are used for plotting the simulation uncertainty bands:
- `acceptable_params_PL_N_dgf_1.csv`
- `acceptable_params_PL_N_dgf_80.csv`
- `acceptable_params_PL_N_dgf_400.csv`

**Best parameter estimation:**
The best parameter set found during parameter estimation for each pDC density has been gathered in a single file. This file is used to plot the main simulation line:
- `final_params.json`

### Results/
This folder is structured to store all simulation plots (`.png`, `.svg`), parameter sets (`.csv`, `.json`), and calculation results (`.txt`) generated by running the scripts. The structure is presented below:

| Folder | Contents |
|---|---|
| **HV** | **PD:** Plots for Figure 2B-D and Supplementary Figure 2<br>**PK:** Plots for Figure 2A and Supplementary Figure 1 |
| **HV_vs_SLE** | **PD:** Plots for Figure 3B & 3D and Supplementary Figure 4<br>**PK:** Plots for Figure 3A & 3C and Supplementary Figure 3 |
| **MCMC** | Plot for Supplementary Figure 10<br>Posterior distribution of parameter sets from MCMC sampling |
| **Parameter_estimation** | The acceptable parameter sets and best parameter set found during parameter estimation |
| **Profile_likelihood** | Plots for Supplementary Figures 7-9<br>Acceptable parameter sets found during PL-analysis<br>95% confidence intervals from PL analysis for Supplementary Table 4-5 |
| **SLE** | **Dosing_protocols:** Plots for Figure 6B and Supplementary Figure 5<br>**PD:** Plots for Figure 5B-C and Supplementary Figure 5D-F<br>**PK:** Plots for Figures 5A & 7, Supplementary Figure 4A-C & 6, skin-to-plasma AUC ratios, and absolute plasma AUC |
| **Validation** | Plots for Figure 4 |

### Scripts/
This folder contains the scripts needed to reproduce the complete model development presented in the article. 

- **`run_parameter_estimation.py`**: Executes the differential evolution algorithm to find the optimal parameter sets.
- **`run_identifiability_analysis.py`**: Performs MCMC sampling and PL analysis. It generates confidence intervals, the acceptable parameter sets used for simulation uncertainty bands and the plots for Supplementary Figures 7-10.
- **`plot_simulations.py`**: Generates all plots for Figures 1-7 and Supplementary Figures 1-6 based on the models, data and parameter sets.
- **`utils.py`**: A helper script containing general functions such as `load_models` and `evaluate_cost` which are imported by the other three scripts.