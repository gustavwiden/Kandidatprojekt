import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sund

# Ladda båda modellerna
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model_HV = sund.load_model("mPBPK_model")
model_SLE = sund.load_model("mPBPK_SLE_model")

# Ladda datafiler
with open("../../../Data/HV_SLE_data.json", "r") as pk_file:
    PK_data = json.load(pk_file)
with open("../../../Data/PD_HV_SLE_data.json", "r") as pd_file:
    PD_data = json.load(pd_file)

# Sätt parametrar
params_HV = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95,
             0.605, 0.2, 5.51, 14.15, 0.28, 2.12e-05, 2.5, 0.525, 0]  # HV
params_SLE = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95,
             0.605, 0.2, 8.91, 14.15, 0.28, 2.12e-05, 2.5, 0.525, 1.27e-5]  # SLE

# Välj dos
dose_key_HV = "IVdose_20_HV"
dose_key_SLE = "IVdose_20_SLE"

# Kroppsvikt
bw = 70

# Funktion för att sätta upp simulering
def setup_sim(data_entry, param_set, model):
    activity = sund.Activity(time_unit='h')
    input_data = data_entry["input"]["IV_in"]
    activity.add_output(sund.PIECEWISE_CONSTANT, "IV_in",
                        t=input_data["t"],
                        f=bw * np.array(input_data["f"]))
    sim = sund.Simulation(models=model, activities=activity, time_unit='h')
    time_vector = np.arange(-10, data_entry["time"][-1] + 6500, 1)
    sim.simulate(time_vector=time_vector, parameter_values=param_set, reset=True)
    return sim

# Simuleringar med olika modeller
sim_HV = setup_sim(PK_data[dose_key_HV], params_HV, model_HV)
sim_SLE = setup_sim(PK_data[dose_key_SLE], params_SLE, model_SLE)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()

# Ändra bakgrundsfärgen för hela figuren
fig.patch.set_facecolor('#fcf5ed')

# Ändra bakgrundsfärgen för axlarna
ax1.set_facecolor('#fcf5ed')
ax2.set_facecolor('#fcf5ed')

# PK-index
idx_pk = sim_HV.feature_names.index("PK_sim")

# Simulerad PK
ax1.plot(sim_HV.time_vector, sim_HV.feature_data[:, idx_pk], '#1b7837', linestyle='--', linewidth=2.5, label="PK sim HV")
ax1.plot(sim_SLE.time_vector, sim_SLE.feature_data[:, idx_pk], '#1b7837', linewidth=2.5, label="PK sim SLE")

# Datapunkter PK
# ax1.errorbar(PK_data[dose_key_HV]["time"], PK_data[dose_key_HV]["BIIB059_mean"],
#              yerr=PK_data[dose_key_HV]["SEM"], fmt='o', color='green', label="PK data HV")
# ax1.errorbar(PK_data[dose_key_SLE]["time"], PK_data[dose_key_SLE]["BIIB059_mean"],
#              yerr=PK_data[dose_key_SLE]["SEM"], fmt='o', color='purple', label="PK data SLE")

# PD-index
idx_pd = sim_HV.feature_names.index("PD_sim")

# Simulerad PD
ax2.plot(sim_HV.time_vector, sim_HV.feature_data[:, idx_pd], '#6d65bf', linestyle='--', linewidth=2.5, label="PD sim HV")
ax2.plot(sim_SLE.time_vector, sim_SLE.feature_data[:, idx_pd], '#6d65bf', linewidth=2.5, label="PD sim SLE")

# Datapunkter PD
# ax2.errorbar(PD_data[dose_key_HV]["time"], PD_data[dose_key_HV]["BDCA2_median"],
#              yerr=PD_data[dose_key_HV]["SEM"], fmt='x', color='green', label="PD data HV")
# ax2.errorbar(PD_data[dose_key_SLE]["time"], PD_data[dose_key_SLE]["BDCA2_median"],
#              yerr=PD_data[dose_key_SLE]["SEM"], fmt='x', color='purple', label="PD data SLE")

# Axlar och etiketter
ax1.set_xlabel("Time [h]", fontsize=22)
ax1.set_ylabel("BIIB059 Plasma Concentration [µg/mL]", color='#1b7837', fontsize=22)
ax1.set_xlim(0, 9000)
ax1.set_ylim(-10, 550)
ax1.spines['left'].set_color('#1b7837')
ax1.tick_params(axis='y', labelcolor='#1b7837', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)

ax2.set_ylabel("BDCA2 Expression [% change from baseline]", color='#6d65bf', fontsize=22)
ax2.set_ylim(-100, 70)
ax2.spines['right'].set_color('#6d65bf')
ax2.tick_params(axis='y', labelcolor='#6d65bf', labelsize=22)

fig.tight_layout()
plt.subplots_adjust(top=1.2)  # Öka från default ca 0.9

# Baseline for bdca2
ax2.axhline(y=0, color='gray', linestyle='dotted', linewidth=2)
ax2.text(60, 1, 'Baseline', color='gray', fontsize=22)



# Add legends
ax1.legend(loc='upper left', fontsize=18, frameon=False)
ax2.legend(loc='upper right', fontsize=18, frameon=False)

# Spara
save_path_svg = "../../../Results/SLE_results/PK_PD/Combined_IVdose_20_HV_vs_SLE.svg"
save_path_png = "../../../Results/SLE_results/PK_PD/Combined_IVdose_20_HV_vs_SLE.png"
os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)

# Spara som SVG
plt.savefig(save_path_svg, format="svg", bbox_inches="tight")

# Spara som PNG
plt.savefig(save_path_png, format="png", bbox_inches="tight", dpi=300)

plt.show()

plt.close()
print(f"Saved plot as SVG to {save_path_svg}")
print(f"Saved plot as PNG to {save_path_png}")