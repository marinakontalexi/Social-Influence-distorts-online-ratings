import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ------------------------------------------------------------------------------
# Function: Load configuration data for a given simulation
# ------------------------------------------------------------------------------
def config(sim, influence):
    """
    Parses simulation data to extract equilibrium values and related statistics.

    Args:
        sim (int): Simulation identifier.
        influence (str): Type of influence setting.

    Returns:
        dict: Configuration dictionary with extracted simulation data.
    """
    eq = [0]  # Default equilibrium starting point
    inf_curve = []  # List to store the social influence curve data
    y = []  # Observed rating curve over time
    m = a_mean = cov = last = -1

    # Find data files
    name = glob(f"../simulations/{influence}/{sim}_experiment*.txt")
    sic = glob(f"../simulations/{influence}/{sim}_sic*.txt")

    # Parse experiment file if it exists
    if name:
        with open(name[0], 'r') as f:
            m, a_mean, cov, last = map(float, f.readline().split('\t'))
        name = name[0].split('_')

    # Parse SIC data if available
    if sic:
        with open(sic[0], 'r') as f:
            eq = []
            for line in f:
                try:
                    # Extract data points from the social influence curve
                    i, j, w, z = line.split('\t')
                    inf_curve.append(float(i))
                    y.append(float(j))
                except ValueError:
                    # Handle cases where parsing fails
                    if len(line.split(" ")) == 1:
                        eq.append(float(line))
                    continue

        name = sic[0].split('_')

    # Return parsed configuration
    return {
        "N": int(name[2]),
        "influence": influence,
        "mean": float(name[3]),
        "alpha": float(name[4]),
        "mod": float(name[5][:-4]),
        "true_mean": float(m),
        "a_mean": float(a_mean),
        "cov": float(cov),
        "last": float(last),
        "equilibria": eq,
        "div": float(m) - eq[0],
        "m": inf_curve,
        "y": y
    }


# ------------------------------------------------------------------------------
# Function: Plot social influence curve
# ------------------------------------------------------------------------------
def plot_inf_curve(plt2, N, mean, alpha, m, y, equilibria, final = [], save=0, ys=[], opaque=0.35):
    """
    Plots the social influence curve.

    Args:
        plt2 (matplotlib.Axes): Target axes for plotting.
        N (int): Population size.
        mean (float): Average rating across the population.
        alpha (float): Alpha value to determine simulation parameter.
        m (list): List of mean influence values.
        y (list): List of observed influence responses.
        equilibria (list): Equilibrium points.
        final (list): List of final observed values.
        save (bool): Whether to save the plot.
        ys (list): List of social influence curve variations for comparison.
        opaque (float): Transparency for visualization.
    """
    # Plot the primary social influence curve
    colors = ["orange", "darkgreen"]
    
    plt2.plot(m, y, color="purple", label=f"alpha = {alpha:.1f}")

    # Overlay comparison influence curves
    for i, (curves, a) in enumerate(ys):
        plt2.plot(m, curves, alpha=opaque, label=f"alpha = {a:.1f}", color=colors[i])

    # Reference line
    plt2.plot(m, m, linestyle='dashed', label="y = x", color="black", alpha=0.7)

    # Mark equilibrium and final observed points
    for eq in equilibria:
        plt2.axvline(x=eq, ymin=0, ymax=eq, linestyle='--', color='gray', alpha=0.5)
    for last in final:
        plt2.axvline(x=last, ymin=0, ymax=last, linestyle='dotted', color='gray')

    # Set plot labels and limits
    plt2.set_xlabel("Observed rating", fontsize=14)
    plt2.set_ylabel(r'Expected rating $\mathbb{E}[R]$', fontsize=14)
    plt2.set_xlim(0, 1)
    plt2.set_ylim(0, 1)
    plt2.legend(fontsize=10)

    # Define x ticks dynamically
    xticks = sorted(list(np.linspace(0, 1, 6)) + equilibria)
    plt2.set_xticks(xticks)
    plt2.set_xticklabels([r'$\star$' if x in equilibria else round(x, 1) for x in xticks])

    return plt2


# ------------------------------------------------------------------------------
# Function: Plot average data
# ------------------------------------------------------------------------------
def plot_avg(ax, N, sims, true_mean, equilibria, save=0):
    """
    Plots average user ratings and overlays equilibrium points.

    Args:
        ax (matplotlib.Axes): Target axes for plotting.
        N (int): Number of users.
        avg (list): List of observeds.
        true_mean (float): Known true average mean value for comparison.
        equilibria (list): List of equilibrium values to overlay on the plot.
        save (bool): Whether to save the plot.
    """
    # Set the x-axis to log scale
    ax.set_xscale("log")
    x = range(1, N + 1)

    # Sort average data according to final average rating
    sims.sort(key=lambda x: x[N - 1])

    norm = plt.Normalize(vmin=min(sims[0][N - 1], 0.2) - true_mean, vmax=max(0.6, sims[-1][N - 1]) - true_mean)

    # Plot averages
    for obs_avg in sims:
        ax.plot(x, obs_avg[:N], color=cm.coolwarm(norm(obs_avg[N - 1] - true_mean)))

    # Plot equilibrium markers
    ax.plot(x, [equilibria[0]]*N, linestyle="dashed", color='gray', label="equilibrium")
    for eq in equilibria[1:]:
        ax.plot(x, [eq]*N, linestyle="dashed", color='gray')

    # Set axis labels and limits
    yticks = sorted(list(np.linspace(0, 1, 6)) + equilibria)
    ax.set_yticks(yticks)
    ax.set_yticklabels([r'$\star$' if x in equilibria else round(x, 1) for x in yticks])
    ax.set_ylabel("Observed rating", fontsize=14)
    ax.set_xlabel("Users", fontsize=14)
    ax.set_xlim(1, N)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)

    # Optionally add a colorbar if multiple averages are being compared
    if len(sims) != 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min(0.2, min(sims, key=lambda x: x[N - 1])[N - 1]),
                                                                      vmax=max(0.6, max(sims, key=lambda x: x[N - 1])[N - 1])))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(r"Observed rating after $10^5$ users", fontsize=12)

    return ax


# ------------------------------------------------------------------------------
# Function: Generate bifurcation plot
# ------------------------------------------------------------------------------
def bifurcation(mean, high, low, alpha, save = 0):
    """
    Generates bifurcation plot visualizing the equilibrium states across parameters.

    Args:
        mean (float): True mean rating value.
        high (list): High equilibrium values.
        low (list): Low equilibrium values.
        alpha (list): Array of alpha values corresponding to equilibrium states.
    """
    # Set up the plot
    f, plt2 = plt.subplots(figsize=(6.5, 4.5), gridspec_kw={'wspace': 0.02}, constrained_layout=True)

    # Normalize for visualization
    norm = plt.Normalize(vmin=-0.7, vmax=0.3)
    c = cm.get_cmap(LinearSegmentedColormap.from_list('blue_purple_red', ["#00008B", "#0000FF", "#800080", "#FF0000"], N=100))

    # Scatter plot visualizations
    for i in range(len(alpha)):
        plt2.scatter(alpha[i], low[i], color=c(norm(low[i] - mean)), s=5)
        plt2.scatter(alpha[i], high[i], color=c(norm(high[i] - mean)), s=5)

    # Add axis labels
    plt2.set_xlabel("Alpha parameter of Beta distribution", fontsize=14)
    plt2.set_ylabel("Equilibria", fontsize=14)
    plt2.set_xlim(0,1.5)
    plt2.set_ylim(0,1)
    plt2.legend(fontsize=12)

    # Add a colorbar to reflect the differences (high - low)
    sm = plt.cm.ScalarMappable(cmap=c, norm=norm)
    sm.set_array([])  # Required for ScalarMappable to work with colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), pad = 0.02)
    cbar.set_label('Distance from true mean rating', fontsize = 13)

    # Finalize and save the visualization
    if save:
        plt.savefig(save, dpi = 300)
    # plt.show()
    plt.close()