from plotting import *
from simulation import *
from glob import glob
from user import *
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Function to compute averages and plot results
# ------------------------------------------------------------------------------
def compute_averages_and_plot(axis, N, S, sim_id, inf, true_mean, main, perc=0):
    """
    Simulates multiple runs, computes the average results, and plots them.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        N (int): Population size.
        S (int): Number of simulations.
        sim_id (int): Starting simulation ID.
        inf (str): Influence type.
        true_mean (float): Known or expected average value.
        main: Main config or precomputed data for plotting.
        perc (int, optional): Percentage adjustment if needed (default=0).

    Returns:
        matplotlib.axes.Axes: Updated axis with the average plots.
    """
    simulation_ids = []

    # Run multiple simulations
    for i in range(S):
        print(f"Running simulation ID: {sim_id + i}")
        # simulation(N, inf, true_mean, 0.3, sim_id + i, flag_sic=0, flag_shuffle=1)
        simulation_ids.append(sim_id + i)

    # Parse simulation results
    averages_data = []
    equilibria = config(main, inf)["equilibria"]

    for sim_id in simulation_ids:
        means = []
        file_path = glob(f'../simulations/{inf}/{sim_id}_experiment*.txt')[0]

        # Read simulation data
        with open(file_path, 'r') as file:
            for line in file:
                data_fields = line.split('\t')
                if len(data_fields) == 3:
                    means.append(float(data_fields[2]))
        averages_data.append(means)
    # Plot computed averages
    return plot_avg(axis, N, averages_data, true_mean, equilibria)


# ------------------------------------------------------------------------------
# Set up simulation parameters and initialize plots
# ------------------------------------------------------------------------------
influence_type = "10"   #  SimpleAvgUser
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
mpl.rcParams['mathtext.default'] = 'regular'

# -------------------------------
# First simulation: Influence Curve
# -------------------------------
N = 10_000
results_y = []
alpha_values = [0.3, 0.1, 3]
simulation_ids = [311, 312, 313]
true_mean = 0.5

# Run simulations with varying alpha values
for idx, sim_id in enumerate(simulation_ids):
    print(f"Processing simulation {sim_id}")
    # simulation(N, influence_type, true_mean, alpha_values[idx], sim_id, flag_exp=0)
    config_data = config(simulation_ids[idx], influence_type)
    results_y.append((config_data["y"], alpha_values[idx]))

# Extract configuration from first simulation
config_data = config(simulation_ids[0], influence_type)

# Plot the first simulation results (influence curve)
ax[0] = plot_inf_curve(
    ax[0], 
    config_data["N"], 
    true_mean, 
    config_data["alpha"], 
    config_data["m"], 
    results_y[0][0], 
    config_data["equilibria"], 
    [],
    f"../plots/{influence_type}/sic_multiple_{simulation_ids[0]}.png", 
    [results_y[1], results_y[2]]
)

print("Influence curve plotting complete.")

# -------------------------------
# Second simulation: Permutation
# -------------------------------
N = 100_000
num_simulations = 100
starting_sim_id = 1700

# Call the averages computation and plot
ax[1] = compute_averages_and_plot(
    ax[1], 
    N, 
    num_simulations, 
    starting_sim_id, 
    influence_type, 
    true_mean, 
    simulation_ids[0]
)

# ------------------------------------------------------------------------------
# Finalize and save the complete plot
# ------------------------------------------------------------------------------
plt.tight_layout()
plt.savefig(f"../plots/{influence_type}/perm_{true_mean:.2f}.png")
