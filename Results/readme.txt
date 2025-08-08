The "Results" folder contains a lot of different plots that are sorted in relevant subfolders, sub-subfolders and so on...

Acceptable params
Contains json and csv files with all acceptable parameter sets found during optimization of the mPBPK_model.txt and mPBPK_SLE_model.txt.
It also contains a json file with the best parameter set that was found for each model and its cost.
It also contains all acceptable parameters found with MCMC sampling for both models.

HV
Contains subfolders with results from mPBPK_model.txt.

    PD
    Contains plots with PD simulations and PD data from phase 1 trial.

    PK
    Contains a plot with PK simulations and PK data from phase 1 trial.
    Also contains a plot of skin vs plasma concentration for comparison with ABC.

SLE
Contains subfolders with results from mPBPK_SLE_model.txt. Divided into "Plasma" and "Skin" to easier find relevant results.

    Plasma
        PD
        Contains plots with PD simulations and PD data from phase 1 trial.

        PK
        Contains a plot with PK simulations and PK data from phase 1 trial.

    Skin
        PD
            Predictions
            Currently empty but will later contain plots which show what our model is capable of predicting with regards to PD.
        
            Parameter sensitivity
                kdegs
                Contains plots which show the sensitivity of changes in the kdegs parameter when simulating PD in skin.

                kints
                Contains plots which show the sensitivity of changes in the kints parameter when simulating PD in skin.
                
                RCS
                Contains plots which show the sensitivity of changes in the RCS parameter when simulating PD in skin.
        
        PK
            Predictions
                Currently empty but will later contain plots which show what our model is capable of predicting with regards to PK.
            
            Parameter sensitivity
                kdegs
                Contains plots which show the sensitivity of changes in the kdegs parameter when simulating PK in skin.

                kints
                Contains plots which show the sensitivity of changes in the kints parameter when simulating PK in skin.
                
                RCS
                Contains plots which show the sensitivity of changes in the RCS parameter when simulating PK in skin.
        
        PKPD 
        Contains plots which show the effect on PK and PD of an increased initial BDCA2 concentration in skin.

Validation
Contains histograms from MCMC-sampling from the optimization of the mPBPK_model.txt and the mPBPK_SLE_model.txt.
Also contains plots of PK simulations and PK data from phase 2A and 2B trials.
Also contains a txt-file with skin/plasma AUC ratios for antibody exposure.
