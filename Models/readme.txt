# Note: Due to occasional CVODE errors, the code might need to be run multiple times. 
# If an error occurs, re-run the code until it succeeds.
# Make sure to be in the right folder to run the code by typing cd *folder* in terminal

// Figure 10: PK plot of HV 
Model: mPBPK_model.text                             // Use output PK_sim 
Optimize: optimize_PK_model.py                      // Make sure to optimize before plotting to get uncertainty
Plot: plot_PK_simulation.py                         // Code might take a while to run due to plotting of uncertainty

// Figure 11: PD plot of HV
Model: mPBPK_model.txt                              // Use output PD_sim 
Optimize: optimize_PD_model.py                      / Make sure to optimize before plotting to get uncertainty
Plot: plot_PD_simulation.py 

// Figure 12: PK/PD plot of both SLE and HV
Models: mPBPK_model.txt & mPBPK_SLE_model.txt        // Use output PK_sim and PD_sim
Plot: plot_HV_SLE_PK_PD.py 

// Figure 13: PK plot of SLE 
Model: mPBPK_SLE_model.txt                           // Use output PK_sim
Optimize: optimize_SLE_PK.py                         // Optimal params will be saved in best_SLE_PK_result.json, where SLE_CL and kmig are the only parameters not fixed
Plot: plot_SLE_PK_simulation.py

// Figure 14: PD plot of SLE
Model: mPBPK_SLE_model.txt                           // Use output PD_sim
Optimize: optimize_SLE_PD.py                         // Optimal params will be saved in best_SLE_PD_result.json, where SLE_CL and kmig are the only parameters not fixed
Plot: plot_SLE_PD_simulation.py

// Figure 15: Plot of SLE PK in skin
Model: mPBPK_SLE_model.txt                           // Use output Skin_PK_sim
Plot: plot_PK_simulation_skin_SLE.py

// Figure 16: Plot of SLE PD in skin
Model: mPBPK_SLE_model.txt                           // Use output Skin_PD_sim
Plot: plot_PD_simulation_skin_SLE.py

