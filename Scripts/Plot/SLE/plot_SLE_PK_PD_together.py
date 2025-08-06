import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

with open("../../../Data/SLE_PK_data.json", "r") as f:
    PK_data = json.load(f)
    
with open("../../../Data/SLE_PD_data.json", "r") as f:
    PD_data = json.load(f)

with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model = sund.load_model("mPBPK_SLE_model")

bodyweight = 69  # Bodyweight in kg

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f=bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

model_sims = {'IVdose_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h')}

time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 2000, 1) for exp in PK_data}

params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.75, 0.95, 0.7467544604963505, 0.2, 0.007326666281272475, 0.9621937056820449, 0.10000000000000002, 5.539999999999999, 5.539999999999999, 2623.9999999999995]


def plot_PK_PD_sim_with_data(params, sims, PK_data, PD_data, time_vectors, save_dir='../../../Results/SLE/Skin/PKPD'):
    os.makedirs(save_dir, exist_ok=True)
    for experiment in PK_data:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # PK simulation (left y-axis)
        ax1.set_xlabel('Time [Hours]')
        ax1.set_ylabel('BIIB059 Skin Concentration (µg/mL)', color='b')
        timepoints = time_vectors[experiment]
        sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
        y = sims[experiment].feature_data[:, 2]
        ax1.plot(timepoints, y, 'b-', label='PK simulation')
        ax1.tick_params(axis='y', labelcolor='b')

        # PD simulation (right y-axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Free BDCA2 Expression % Change from Baseline', color='r')
        y = sims[experiment].feature_data[:, 3]
        ax2.plot(timepoints, y, 'r-', label='PD simulation')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends
        ax1.legend(bbox_to_anchor=(0.95, 0.60))
        ax2.legend(bbox_to_anchor=(0.95, 0.50))

        plt.title(f"PK and PD Simulation for {experiment} with 179 pDCs/mm²")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{experiment}_PK_PD_sim_179_pDC_mm2.png")
        plt.savefig(save_path, dpi=600)
        plt.show()

plot_PK_PD_sim_with_data(params, model_sims, PK_data, PD_data, time_vectors)
