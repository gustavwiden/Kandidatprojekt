import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import the base directory logic from your existing utils file
from utils import base_dir

# 1. Define Directories using utils
output_dir = os.path.join(base_dir, 'Results', 'MCMC')
csv_path = os.path.join(output_dir, 'MCMC_sampling_result_model_backup.csv')
ci_path = os.path.join(base_dir, 'Results', 'Profile_likelihood', '95_CI_Bounds_80.txt')

# 2. Check for required files
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find the MCMC backup file at: {csv_path}")
if not os.path.exists(ci_path):
    raise FileNotFoundError(f"Could not find the confidence interval file at: {ci_path}")

print(f"Loading MCMC trace from: {csv_path}...")
trace_array = np.loadtxt(csv_path, delimiter=',')
print("Data loaded successfully. Plotting...")

# 3. Define Parameter Names (Manually extracted from your original script)
parameter_names = ['F', 'ka', 'RC2', 'CL_HV', 'CL_SLE', 'kdeg']

# 4. Parse confidence intervals from the profile likelihood file
ci_bounds = {}
with open(ci_path, 'r', encoding='utf-8') as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line.startswith('95%') or line.startswith('='):
            continue
        if ':' not in line:
            continue
        name, value_text = line.split(':', 1)
        name = name.strip()
        value_text = value_text.strip()
        values = value_text.strip('[]').split(',')
        if len(values) == 2:
            lower, upper = [float(v.strip()) for v in values]
            ci_bounds[name] = (lower, upper)

# 5. The Original Plotting Logic
rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(14, 7))
axes = axs.flatten()
num_params = trace_array.shape[1]
color = plt.cm.Blues(np.linspace(0.85, 0.85, 1))
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.35)

for i in range(num_params):
    ax = axes[i]
    data_for_hist = trace_array[:, i]
    ax.hist(data_for_hist, bins='auto', color=color[0])

    if parameter_names[i] in ci_bounds:
        lower, upper = ci_bounds[parameter_names[i]]
        lower_line = ax.axvline(lower, linestyle=':', color='black', linewidth=1.5, alpha=0.7)
        upper_line = ax.axvline(upper, linestyle=':', color='black', linewidth=1.5, alpha=0.7)

        span = upper - lower
        pad = 0.25 * span
        ax.set_xlim(lower - pad, upper + pad)

    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xlabel(f'{parameter_names[i]} Value', fontsize=14)

    if parameter_names[i] in ci_bounds:
        ax.text(
            0.5,
            0.92,
            '95% Confidence Interval',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
            color='black',
        )
        ax.annotate(
            '',
            xy=(0.82, 0.925),
            xytext=(0.76, 0.925),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2, shrinkA=0, shrinkB=0),
        )
        ax.annotate(
            '',
            xy=(0.18, 0.925),
            xytext=(0.24, 0.925),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2, shrinkA=0, shrinkB=0),
        )

    formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelrotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_ylim(0, 13000)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000, 12000])

# Remove any unused subplots
for j in range(num_params, len(axes)):
    fig.delaxes(axes[j])

# 6. Save and Show
save_dir = os.path.join(base_dir, 'Results', 'MCMC')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "MCMC_mPBPK-model_model_test.svg")

plt.tight_layout()
plt.savefig(save_path, format='svg')
print(f"Plot saved to: {save_path}")
plt.show()