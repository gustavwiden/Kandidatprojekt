import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sund

# Ladda båda modellerna
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model_SLE = sund.load_model("mPBPK_SLE_model")

# Ladda datafiler
with open("../../../Data/SLE_PK_data.json", "r") as pk_file:
    PK_data = json.load(pk_file)
with open("../../../Data/SLE_PD_data.json", "r") as pd_file:
    PD_data = json.load(pd_file)

# Sätt parametrar
params_SLE = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95, 
0.605, 0.2, 8.93, 14.7, 0.274, 1.635e-05, 2.2, 0.466, 6.54e-6]  # SLE

# Kroppsvikt
bw = 70

# Funktion för att sätta upp simulering
def setup_sim(data_entry, param_set, model):
    activity = sund.Activity(time_unit='h')
    # Kolla vilken input som finns
    if "IV_in" in data_entry["input"]:
        input_key = "IV_in"
        dose_f = bw * np.array(data_entry["input"][input_key]["f"])  # multiplicera med bw för IV
    elif "SC_in" in data_entry["input"]:
        input_key = "SC_in"
        dose_f = np.array(data_entry["input"][input_key]["f"])       # ingen multiplikation för SC
    else:
        raise KeyError("Varken 'IV_in' eller 'SC_in' finns i input-data!")
    input_data = data_entry["input"][input_key]
    activity.add_output(sund.PIECEWISE_CONSTANT, input_key,
                        t=input_data["t"],
                        f=dose_f)
    sim = sund.Simulation(models=model, activities=activity, time_unit='h')
    time_vector = np.arange(-10, data_entry["time"][-1] + 6500, 1)
    sim.simulate(time_vector=time_vector, parameter_values=param_set, reset=True)
    return sim

# Exempel på axelinställningar per dos
axes_settings = {
    "IVdose_005_HV": {"xlim": (0, 3000), "ylim_pk": (-0.001, 0.07), "ylim_pd": (-12, 5)},
    "IVdose_03_HV":  {"xlim": (0, 4000), "ylim_pk": (-0.001, 0.6), "ylim_pd": (-65, 10)},
    "IVdose_1_HV":   {"xlim": (0, 6000), "ylim_pk": (-0.001, 1.9), "ylim_pd": (-85, 20)},
    "IVdose_3_HV":   {"xlim": (0, 7000), "ylim_pk": (-0.001, 6), "ylim_pd": (-100, 45)},
    "IVdose_20_SLE": {"xlim": (0, 8500), "ylim_pk": (-1, 40), "ylim_pd": (-100, 45)},
    "SCdose_50_HV":  {"xlim": (0, 6000), "ylim_pk": (-0.001, 0.8), "ylim_pd": (-65, 20)},
    # Lägg till fler doser vid behov
}

with open("../../../Results/Acceptable params/acceptable_params_SLE_PK.json", "r") as f:
    acceptable_params_PK = json.load(f)
acceptable_params_PK = [np.array(p) for p in acceptable_params_PK]

with open("../../../Results/Acceptable params/acceptable_params_SLE_PD.json", "r") as f:
    acceptable_params_PD = json.load(f)
acceptable_params_PD = [np.array(p) for p in acceptable_params_PD]

# Loopa över alla doser i PK_data
for dose_key_SLE in PK_data.keys():
    print(f"Plottar för dos: {dose_key_SLE}")

    # Simulering för aktuell dos
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
    idx_pk = sim_SLE.feature_names.index("Skin_PK_sim")

    # Simulerad PK
    ax1.plot(sim_SLE.time_vector, sim_SLE.feature_data[:, idx_pk], '#1b7837', linewidth=2.5, label="PK sim SLE")

    # PD-index
    idx_pd = sim_SLE.feature_names.index("Skin_PD_sim")

    # Simulerad PD
    ax2.plot(sim_SLE.time_vector, sim_SLE.feature_data[:, idx_pd], '#6d65bf', linewidth=2.5, label="PD sim SLE")

        # PK uncertainty band
    if acceptable_params_PK:
        y_min_pk = np.full_like(sim_SLE.time_vector, np.inf, dtype=float)
        y_max_pk = np.full_like(sim_SLE.time_vector, -np.inf, dtype=float)
        for params in acceptable_params_PK:
            sim_SLE.simulate(time_vector=sim_SLE.time_vector, parameter_values=params, reset=True)
            y_sim = sim_SLE.feature_data[:, idx_pk]
            y_min_pk = np.minimum(y_min_pk, y_sim)
            y_max_pk = np.maximum(y_max_pk, y_sim)
        # Only add label for the first dose to avoid duplicate legend entries
        if dose_key_SLE == list(PK_data.keys())[0]:
            ax1.fill_between(sim_SLE.time_vector, y_min_pk, y_max_pk, color='#1b7837', alpha=0.18, label="PK uncertainty SLE")
        else:
            ax1.fill_between(sim_SLE.time_vector, y_min_pk, y_max_pk, color='#1b7837', alpha=0.18)

    # PD uncertainty band
    if acceptable_params_PD:
        y_min_pd = np.full_like(sim_SLE.time_vector, np.inf, dtype=float)
        y_max_pd = np.full_like(sim_SLE.time_vector, -np.inf, dtype=float)
        for params in acceptable_params_PD:
            sim_SLE.simulate(time_vector=sim_SLE.time_vector, parameter_values=params, reset=True)
            y_sim = sim_SLE.feature_data[:, idx_pd]
            y_min_pd = np.minimum(y_min_pd, y_sim)
            y_max_pd = np.maximum(y_max_pd, y_sim)
        if dose_key_SLE == list(PK_data.keys())[0]:
            ax2.fill_between(sim_SLE.time_vector, y_min_pd, y_max_pd, color='#6d65bf', alpha=0.18, label="PD uncertainty SLE")
        else:
            ax2.fill_between(sim_SLE.time_vector, y_min_pd, y_max_pd, color='#6d65bf', alpha=0.18)

    # Axlar och etiketter
    ax1.set_xlabel("Time [h]", fontsize=22)
    ax1.set_ylabel("BIIB059 Skin Concentration [µg/mL]", color='#1b7837', fontsize=22)
    ax1.spines['left'].set_color('#1b7837')
    ax1.tick_params(axis='y', labelcolor='#1b7837', labelsize=22)
    ax1.tick_params(axis='x', labelsize=22)

    ax2.set_ylabel("BDCA2 Expression [% change from baseline]", color='#6d65bf', fontsize=22)
    ax2.spines['right'].set_color('#6d65bf')
    ax2.tick_params(axis='y', labelcolor='#6d65bf', labelsize=22)

    # Hämta axelinställningar för aktuell dos, annars default
    settings = axes_settings.get(dose_key_SLE, {"xlim": (0, 9000), "ylim_pk": (-10, 50), "ylim_pd": (-100, 45)})
    ax1.set_xlim(*settings["xlim"])
    ax1.set_ylim(*settings["ylim_pk"])
    ax2.set_ylim(*settings["ylim_pd"])

    fig.tight_layout()
    plt.subplots_adjust(top=1.2)  # Öka från default ca 0.9

    # Baseline for bdca2
    ax2.axhline(y=0, color='#6d65bf', linestyle='dotted', linewidth=2)
    ax2.text(settings["xlim"][1] * 0.88, 1, 'Baseline', color='#6d65bf', fontsize=16)

    # # Treshold 1 microgram/mL   
    # ax1.axhline(y=1, color='gray', linestyle='dotted', linewidth=1)
    # #ax1.text(60, 1, 'Baseline', color='gray', fontsize=22)

    # Add legends
    ax1.legend(loc='upper left', fontsize=18, frameon=False)
    ax2.legend(loc='upper right', fontsize=18, frameon=False)

    # Spara
    save_path_svg = f"../../../Results/Skin_SLE/PK_PD/Combined_{dose_key_SLE}.svg"
    save_path_png = f"../../../Results/Skin_SLE/PK_PD/Combined_{dose_key_SLE}.png"
    os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)

    # Spara som SVG
    plt.savefig(save_path_svg, format="svg", bbox_inches="tight")

    # Spara som PNG
    plt.savefig(save_path_png, format="png", bbox_inches="tight", dpi=300)

    # plt.show()

    plt.close()
    print(f"Saved plot as SVG to {save_path_svg}")
    print(f"Saved plot as PNG to {save_path_png}")