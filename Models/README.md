# Models Overview

The "Models" folder contains `.txt` files according to the SUND model format.

## SLE Patient Models

There are three models representing SLE patients.
These reflect different degrees of lesional pDC infiltration and thus BDCA target burden:
* [`SLE_model_1.txt`](SLE_model_1.txt) (1 pDC/mm² in skin)
* [`SLE_model_80.txt`](SLE_model_80.txt) (80 pDCs/mm² in skin)
* [`SLE_model_400.txt`](SLE_model_400.txt) (400 pDCs/mm² in skin)

## Healthy Volunteer Models

There are two models representing healthy volunteers.
These reflect the normal and high-end pDC plasma concentration:
* [`HV_model.txt`](HV_model.txt) (5100 pDCs/L in plasma)
* [`HV_model_high.txt`](HV_model_high.txt)* (10200 pDCs/L in plasma)

*Only used in plot_skin_plasma_concentration_ratio in [`plot_skin_simulations.py`](../Scripts/plot_skin_simulations.py) to generate Figure 7C.