# Data Files Overview

This folder contains all `.json` files needed to run the scripts.

## Digitised Literature Data

The following files are from the **phase I trial**:
* [HV_PK_data.json](HV_PK_data.json)
* [HV_PD_data.json](HV_PD_data.json)
* [SLE_PK_data.json](SLE_PK_data.json)
* [SLE_PD_data.json](SLE_PD_data.json)

**Used in:**
* [`plot_plasma_simulations.py`](../Scripts/plot_plasma_simulations.py)
* [`run_Parameter_estimation.py`](../Scripts/run_parameter_estimation.py)
* [`run_mcmcm_pl.py`](../Scripts/run_mcmcm_pl.py)
* [`utils.py`](../Scripts/utils.py)

---

The following file is from the **phase II trial**:
* [SLE_CLE_PK_validation_data.json](SLE_CLE_PK_validation_data.json)

**Used in:**
* [`plot_plasma_simulations.py`](../Scripts/plot_plasma_simulations.py)
* [`plot_skin_simulations.py`](../Scripts/plot_skin_simulations.py)
* [`utils.py`](../Scripts/utils.py)

---

## Generated Data

Additional `.json` files created when running [`plot_plasma_simulations.py`](../Scripts/plot_plasma_simulations.py) and [`plot_skin_simulations.py`](../Scripts/plot_skin_simulations.py):

* [HV_vs_SLE_plasma_PD_response_data.json](HV_vs_SLE_plasma_PD_response_data.json)
* [HV_vs_SLE_plasma_PK_response_data.json](HV_vs_SLE_plasma_PK_response_data.json)
* [SLE_IV_dose_skin_PD_response_data.json](SLE_IV_dose_skin_PD_response_data.json)
* [SLE_skin_SC_clinical_menu_loading_1.json](SLE_skin_SC_clinical_menu_loading_1.json)
* [SLE_skin_SC_clinical_menu_loading_80.json](SLE_skin_SC_clinical_menu_loading_80.json)
* [SLE_skin_SC_clinical_menu_loading_400.json](SLE_skin_SC_clinical_menu_loading_400.json)
* [SLE_skin_SC_min_maintenance_1.json](SLE_skin_SC_min_maintenance_1.json)
* [SLE_skin_SC_min_maintenance_80.json](SLE_skin_SC_min_maintenance_80.json)
* [SLE_skin_SC_min_maintenance_400.json](SLE_skin_SC_min_maintenance_400.json)