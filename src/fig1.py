import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from user import *
import statistics as stat

def plot_linear_sic(mean, r, a_values):
    """
    Plots the expected rating against the observed average rating.

    Parameters:
    - mean: Array of observed average ratings.
    - r: True mean rating divided by 5.
    - a_values: List of tuples containing influence values and covariance.
    """
    maps = "viridis"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'wspace': 0.03}, constrained_layout=True)
    cmap = cm.get_cmap(maps)

    for j, ax in enumerate(axes):
        min_eq = 1
        max_eq = 0
        values = a_values[j]
        for a, cov in values:
            f_mean = a * mean + (1 - a) * r - cov  # Expected rating
            eq = r - cov / (1 - a)  # Equilibrium point
            ax.plot(mean, f_mean, color=cmap(a), linewidth=1)
            min_eq = min(min_eq, eq)
            max_eq = max(max_eq, eq)

        # Add reference lines
        ax.vlines(x=r, ymin=0, ymax=r, color='black', linestyle=':', linewidth=1, label='True mean rating')
        ax.plot(mean, mean, color='gray', linestyle='--', linewidth=1, label='y = x')

        # Customize ticks and limits
        xticks = sorted([x / 5 for x in range(0, 6)] + [r])
        ax.set_xticks(xticks)
        ax.set_xticklabels([r'$\star$' if x == r else ("0" if x == 0 else ("1" if x == 1 else x)) for x in xticks])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Observed rating", fontsize=14)

    # Highlight possible equilibria
    axes[1].fill_between(mean, mean, np.zeros(len(mean)), 
                         where=(mean >= np.floor(min_eq * 100) / 100) & (mean <= np.ceil(max_eq * 100) / 100), 
                         color='lightgray', alpha=0.3, label="Possible equilibria")

    # Add labels and legend
    axes[0].set_ylabel(r'Expected rating $\mathbb{E}[R]$', fontsize=14)
    axes[1].legend(fontsize=10, loc="upper left")
    axes[1].set_yticks([])

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=maps, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label(r'Mean influence $\mathbb{E}[\lambda]$', fontsize=14)

    # Save and display the plot
    plt.savefig('../plots/00/sic_linear.png', dpi=300)
    # plt.show()

# Ensure proper rendering of math symbols
mpl.rcParams['mathtext.default'] = 'regular'

# Define parameters
obs_avg = np.linspace(0, 1, 100)  # Range of observed average ratings
R = 3.75  # True mean rating
r = R / 5  # Normalized true mean rating
lamda = np.linspace(0, 0.99, 20)  # Range of influence values
moderators = [0, 0.1, 0.15] + [max(0, min(1, np.random.normal(0.7, 0.3))) for _ in range(17)]
N = 1000  # Number of users

# Prepare influence values and covariances
a_values = [[], []]

# Generate influence values for the first subplot
for l in lamda:
    a_values[0].append((l, 0))

# Generate influence values and covariances for the second subplot
for a in moderators:
    np.random.seed(1)
    users = [ModeratedUser(r, alpha=3, moderator=a) for _ in range(N)]
    true_mean = stat.mean([x.rating for x in users])
    a_mean = stat.mean([x.Lambda() for x in users])
    cov = np.cov([x.rating for x in users], [x.Lambda() for x in users])[0, 1]
    a_values[1].append((a_mean, cov))

# Generate the plot
plot_linear_sic(obs_avg, r, a_values)
