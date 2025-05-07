import matplotlib.pyplot as plt
import numpy as np

def plot_PK_PD_sim_with_data(params, sims, experiment, PK_data, PD_data, time_vectors, feature_to_plot_PK='PK_sim', feature_to_plot_PD='PD_sim'):
    """
    Plots both PK_sim and PD_sim in the same figure with dual y-axes for a single experiment.
    
    Left y-axis: BIIB059 concentration (PK_sim)
    Right y-axis: BDCA2 % change from baseline (PD_sim)
    
    Parameters:
        params: List of parameters for the simulation.
        sims: Dictionary of simulations for PK and PD.
        experiment: The name of the experiment to plot.
        PK_data: Dictionary containing PK experimental data.
        PD_data: Dictionary containing PD experimental data.
        time_vectors: Dictionary of time vectors for each experiment.
        feature_to_plot_PK: Feature name for PK simulation.
        feature_to_plot_PD: Feature name for PD simulation.
    """
    if experiment not in PD_data:
        print(f"Skipping {experiment} as it is not present in both PK and PD data.")
        return None

    # Use the experiment name directly as the legend label
    legend_label = experiment

    # Create a new figure
    fig, ax1 = plt.subplots()

    # Plot PK_sim on the left y-axis
    ax1.set_xlabel('Time [Hours]')
    ax1.set_ylabel('BIIB059 Concentration (µg/mL)', color='b')
    timepoints = time_vectors[experiment]

    # Simulate and plot PK data
    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx_PK = sims[experiment].feature_names.index(feature_to_plot_PK)
    ax1.plot(sims[experiment].time_vector, sims[experiment].feature_data[:, feature_idx_PK], 'b-')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for PD_sim
    ax2 = ax1.twinx()
    ax2.set_ylabel('BDCA2 Expression % Change from Baseline', color='r')

    # Simulate and plot PD data
    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx_PD = sims[experiment].feature_names.index(feature_to_plot_PD)
    ax2.plot(sims[experiment].time_vector, sims[experiment].feature_data[:, feature_idx_PD], 'r-')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a title
    plt.title(f"PK and PD simulation for {experiment}")

    # Adjust layout
    plt.tight_layout()

    # Return the figure object for saving
    return fig

