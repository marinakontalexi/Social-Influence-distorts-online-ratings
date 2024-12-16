# Import necessary modules and functions
from plotting import *
from social_influence_experimental import *
from user import *
import matplotlib as mpl
import numpy as np


# ------------------------------------------------------------------------------
# Function to compute bifurcation points
# ------------------------------------------------------------------------------
def compute_bifurcation(sim_id, sim_ids_list, N, true_mean, A):
    """
    Simulate and analyze bifurcation behavior by computing high and low equilibrium values.

    Args:
        sim_id (int): Starting simulation ID.
        sim_ids_list (list[int]): List of simulation IDs.
        N (int): Population size for simulations.
        m (float): True mean rating.
        A (list[float]): Array of alpha values to investigate.
    """
    num_alphas = len(A)  # Number of alpha values to simulate over
    high = np.zeros(num_alphas)  # Array to store high equilibrium values
    low = np.zeros(num_alphas)   # Array to store low equilibrium values

    # Loop over alpha values and compute equilibria
    for idx, alpha_value in enumerate(A):
        if not sim_ids_list:  # If no simulation IDs are provided
            simulation(N, influence, true_mean, alpha_value, sim_id + idx, flag_exp=0)
            equilibria = config(sim_id + idx, influence)["equilibria"]
        else:  # Use provided simulation IDs
            equilibria = config(sim_ids_list[idx], influence)["equilibria"]

        # Record the highest and lowest equilibrium points
        high[idx] = equilibria[-1]  # Last equilibrium value
        low[idx] = equilibria[0]   # First equilibrium value

    # Call the bifurcation plotting function
    bifurcation(true_mean, high, low, A, f"../plots/10/bifurcation_{true_mean:.1f}_cm.png")


# ------------------------------------------------------------------------------
# Set up matplotlib settings and simulation parameters
# ------------------------------------------------------------------------------
mpl.rcParams['mathtext.default'] = 'regular'  # Ensure math text is rendered normally
influence = "10"  #  SimpleAvgUser

# Set simulation parameters
sim_id = 991  # Starting simulation ID
N = 10**5  # Population size
true_mean = 0.7  # Mean influence
accuracy = 210  # Number of simulations to determine bifurcation
sims = range(sim_id, sim_id + accuracy)  # Simulation ID range
alpha_values = []

# Extract alpha parameter values from simulations
if sims == []:
    alpha_values = np.linspace(0,1,accuracy)
for s_id in sims:
    config_data = config(s_id, influence)
    alpha_values.append(config_data["alpha"])

compute_bifurcation(sim_id, sims, N, true_mean, alpha_values)
print("Bifurcation analysis and visualization complete.")
