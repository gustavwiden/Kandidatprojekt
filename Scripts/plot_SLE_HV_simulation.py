# Importing the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import sund
import json

# JSON encoder for saving numpy arrays
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Load model definition
model_path = "../Models/mPBPK_SLE_model.txt"
sund.install_model(model_path)
first_model = sund.load_model("mPBPK_SLE_model")

# Load PK data
with open("../Data/HV_SLE_data.json", "r") as f:
    HV_SLE_data = json.load(f)


# Setup parameters
params_HV = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.769, 0.95, 0.605,
             0.2, 5.5, 16356, 336, 1.31e-1, 8, 525, 0.0001] # HV_CL

params_SLE = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.769, 0.95, 0.605,
              0.2, 10.43, 20900, 281, 1.31e-1, 8, 525, 0.07] # SLE_CL

# Bodyweight
bodyweight = 70

# Create activity
IV_20_HV = sund.Activity(time_unit='h')
IV_20_SLE = sund.Activity(time_unit='h')

# Create simulation 
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",
                   t=HV_SLE_data['IVdose_20_HV']['input']['IV_in']['t'],
                   f=bodyweight * np.array(HV_SLE_data['IVdose_20_HV']['input']['IV_in']['f']))

IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",
                     t=HV_SLE_data['IVdose_20_SLE']['input']['IV_in']['t'],
                     f=bodyweight * np.array(HV_SLE_data['IVdose_20_SLE']['input']['IV_in']['f']))


sim_HV = sund.Simulation(models=first_model, activities=IV_20_HV, time_unit='h')
sim_SLE = sund.Simulation(models=first_model, activities=IV_20_SLE, time_unit='h')


# Time vectors
time_vector_HV = np.arange(-10, HV_SLE_data['IVdose_20_HV']['time'][-1] + 0.1, 1)
time_vector_SLE = np.arange(-10, HV_SLE_data['IVdose_20_SLE']['time'][-1] + 0.1, 1)

# Run simulations
sim_HV.simulate(time_vector=time_vector_HV, parameter_values=params_HV, reset=True)
sim_SLE.simulate(time_vector=time_vector_SLE, parameter_values=params_SLE, reset=True)

# Plotting
feature_to_plot = 'PK_sim'
feature_idx_HV = sim_HV.feature_names.index(feature_to_plot)
feature_idx_SLE = sim_SLE.feature_names.index(feature_to_plot)

plt.figure(figsize=(12, 7))

# Simuleringskurvor
plt.plot(sim_HV.time_vector, sim_HV.feature_data[:, feature_idx_HV],
         label='IV 20 HV simulation', color='green', linestyle='-', linewidth=2)

plt.plot(sim_SLE.time_vector, sim_SLE.feature_data[:, feature_idx_SLE],
         label='IV 20 SLE simulation', color='purple', linestyle='-', linewidth=2)


# Datapunkter
plt.errorbar(HV_SLE_data['IVdose_20_HV']['time'], HV_SLE_data['IVdose_20_HV']['BIIB059_mean'],
             yerr=HV_SLE_data['IVdose_20_HV']['SEM'], fmt='o', color='green',
             label='IV 20 HV data', capsize=3)

plt.errorbar(HV_SLE_data['IVdose_20_SLE']['time'], HV_SLE_data['IVdose_20_SLE']['BIIB059_mean'],
             yerr=HV_SLE_data['IVdose_20_SLE']['SEM'], fmt='o', color='purple',
             label='IV 20 SLE data', capsize=3)

plt.xlabel('Time [Hours]', fontsize=14)
plt.ylabel('BIIB059 concentration (µg/ml)', fontsize=14)
plt.title('PK simulation: IV 20 mg/kg dose in HV vs SLE', fontsize=18)
plt.yscale('log')
plt.ylim(0.1, 700)
plt.xlim(-25, 3000)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
os.makedirs('../Results', exist_ok=True)
save_path = '../Results/PK_SLE_HV_20.png'
plt.savefig(save_path, bbox_inches='tight')
plt.close()

