The "Scripts" folder contains all scripts used during the project, sorted in relevant subfolders.

Optimize
Contains the two scripts used for optimization of mPBPK_model.txt and mPBPK_SLE_model.txt.

Plot
Contains two sub-subfolders depedning on if the plotting-script is related to HV or SLE

    HV
    Contains the scripts to plot all figures in Results/HV
    plot_PK_simulation.py and plot_ABC_plasma_skin.py are used to generate figures in Results/HV/PK
    plot_PD_simulation.py is used to generate figures in Result/HV/PD

    SLE
    Contains the scripts to plot all figures in Results/SLE
    plot_SLE_PD_simulation.py is used to generate figures in Results/SLE/Plasma/PD
    plot_SLE_PK_simulation.py is used to generate figures in Results/SLE/Plasma/PK
    plot_SLE_PK_PD_skin_uncertainty.py is used to generate figures in Results/SLE/Skin/PK and ../../../PD 
    plot_SLE_PK_PD_skin_together.py is used to generate figures in Results/SLE/Skin/PKPD

Validation
Contains the scripts to plot all figures in Results/Validation
calculate_AUC_skin_plasma.py is used to generate AUC_ratios.txt
MCMC_SLE.py is used to generate the histogram MCMC SLE mPBPK-model.png
MCMC.py is used to generate the histogram MCMC mPBPK-model.png
plot_validation_CLE.py is used to generate the validation plots of PK with CLE data from phase 2B
plot_validation_SLE.py is used to generate the validation plots of PK with SLE data from phase 2A



