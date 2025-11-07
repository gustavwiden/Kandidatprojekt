# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sund
import json

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model_32_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Load acceptable parameters for mPBPK_model
with open("../../../Results/Acceptable params/acceptable_params_SLE_32_pdc_mm2.json", "r") as f:
    acceptable_params = json.load(f)

# Load final parameters for mPBPK_model
with open("../../../Results/Acceptable params/best_SLE_result_32_pdc_mm2.json", 'r') as f:
    params = np.array(json.load(f)['best_param'])

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model_32_pdc_mm2.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model_32_pdc_mm2")

# Average bodyweight for SLE patient
bodyweight = 70

# Creating activity objects for each dose
IV_005_SLE = sund.Activity(time_unit='h')
IV_005_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 50]))

IV_03_SLE = sund.Activity(time_unit='h')
IV_03_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 300]))

IV_1_SLE = sund.Activity(time_unit='h')
IV_1_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 1000]))

IV_3_SLE = sund.Activity(time_unit='h')
IV_3_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 3000]))

IV_10_SLE = sund.Activity(time_unit='h')
IV_10_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 10000]))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 20000]))

IV_40_SLE = sund.Activity(time_unit='h')
IV_40_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 40000]))

IV_60_SLE = sund.Activity(time_unit='h')
IV_60_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 60000]))

# Creating simulation objects for each dose
model_sims = {
            'IV_005_SLE': sund.Simulation(models=model, activities=IV_005_SLE, time_unit='h'),
            'IV_03_SLE': sund.Simulation(models=model, activities=IV_03_SLE, time_unit='h'),
            'IV_1_SLE': sund.Simulation(models=model, activities=IV_1_SLE, time_unit='h'),
            'IV_3_SLE': sund.Simulation(models=model, activities=IV_3_SLE, time_unit='h'),
            'IV_10_SLE': sund.Simulation(models=model, activities=IV_10_SLE, time_unit='h'),
            'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
            'IV_40_SLE': sund.Simulation(models=model, activities=IV_40_SLE, time_unit='h'),
            'IV_60_SLE': sund.Simulation(models=model, activities=IV_60_SLE, time_unit='h')}

# Define time vectors for each dose
time_vector_low_IV_doses = np.arange(-10, 200, 1)
time_vector_medium_IV_doses = np.arange(-10, 2000, 1)
time_vector_high_IV_doses = np.arange(-10, 4600, 1)

# Define a function to plot the ratio of anitbody concentration in skin and plasma for all doses
# Compare this value with the general ABC ratio of 0.157 by Shah and Betts (2012)
def plot_ABC_plasma_vs_skin(selected_params, sims, time_vectors, save_dir='../../../Results/SLE/Skin/PK'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    # Define colors and labels for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#a0e78e', "#afe78e"]
    doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
    dose_labels = ['0.05 mg/kg IV', '0.3 mg/kg IV', '1 mg/kg IV', '3 mg/kg IV', '10 mg/kg IV', '20 mg/kg IV', '40 mg/kg IV', '60 mg/kg IV']

    # Create empty lists to store all skin and plasma concentrations
    all_skin = []
    all_plasma = []

    # Iterate through each dose and simulate PK in plasma and skin
    for (dose, color, label) in (zip(doses, colors, dose_labels)):

        if dose in ('IV_005_SLE', 'IV_03_SLE'):
            time_vector = time_vectors['low_dose']
        elif dose in ('IV_1_SLE', 'IV_3_SLE'):
            time_vector = time_vectors['medium_dose']
        else:
            time_vector = time_vectors['high_dose'] 

        sims[dose].simulate(time_vector=time_vector, parameter_values=selected_params, reset=True)
        skin = sims[dose].feature_data[:, 2]
        plasma = sims[dose].feature_data[:, 0]

        # Only keep points where both values are positive
        valid = (skin > 0) & (plasma > 0)
        skin = skin[valid]
        plasma = plasma[valid]

        # Plot the skin vs plasma concentration
        plt.scatter(plasma, skin, color=color, label=label, alpha=0.7)
        all_skin.extend(skin)
        all_plasma.extend(plasma)

    # Calculate the minimum and maximum values for skin and plasma concentrations
    x_min = min(all_plasma)
    x_max = max(all_plasma)
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)

    # Plot the reference lines for the general ABC ratio
    plt.plot(x_fit, 0.157 * x_fit, 'k-', linewidth=2, label='Skin ABC (15.7 %)')
    plt.plot(x_fit, 0.0785 * x_fit, 'k--', linewidth=2, label='2-fold Error')
    plt.plot(x_fit, 0.314 * x_fit, 'k--', linewidth=2)

    # Set scale, labels, title, and legend
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    plt.ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize=18)
    plt.title('Model Simulations vs Antibody Biodistribution Coefficient', fontsize=22)
    plt.legend(fontsize=15, loc='lower right', ncols=2)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Save the plot
    save_path = os.path.join(save_dir, "Skin vs Plasma Concentration.png")
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

time_vectors_IV = {'low_dose': time_vector_low_IV_doses, 'medium_dose': time_vector_medium_IV_doses, 'high_dose': time_vector_high_IV_doses}

# Run the function to plot the skin vs plasma concentration for all doses
plot_ABC_plasma_vs_skin(params, model_sims, time_vectors_IV)

