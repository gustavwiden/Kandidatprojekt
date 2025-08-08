The "Data" folder cotains all json-files used during model development.

PHASE 1
PK_data and PD_data are from HV in the phase 1 trial. Used for optimization of mPBPK_model.txt.

SLE_PK_data and SLE_PD_data are from SLE patients in the phase 1 trial. Used for optimization of mPBPK_SLE_model.txt.

SLE_PK_data_plotting and SLE_PD_data_plotting are just used to get relevant timevectors for plotting simulations
in SLE patients if they were given doses that were only given to HV in the phase 1 trial (all except 20 mg/kg).


PHASE 2A
SLE_Validation_PK_data are from SLE patients in the phase 2A trial. Used for validation of mPBPK_SLE_model.txt.

PHASE 2B
CLE_Validation_PK_data are from CLE patients (with or without concomitant SLE) in the phase 2B trial.
Also used for validation of mPBPK_SLE_model.txt, however with the knowledge that there are differences between
SLE and CLE that are not accounted for in the model.


