from user import *
import numpy as np
import random

# Runs simulations to model user behavior and social influence
def simulation(N, inf, M, alpha, sim_id, moderator=0, flag_exp=1, flag_sic=1, empirical_sic=False, flag_shuffle=0, fixed_perc=0): 
    """
    Simulates user behavior and computes social influence curves.

    Parameters:
        N (int): Number of users.
        inf (str): Influence type (e.g., "00", "01", "10", etc.).
        M (float): True mean rating
        alpha (float): Beta distribution parameter.
        sim_id (int): Simulation ID for file naming.
        moderator (float): Moderator's influence.
        flag_exp (bool): Whether to perform the experiment.
        flag_sic (bool): Whether to compute the social influence curve.
        empirical_sic (bool): Use empirical SIC computation.
        flag_shuffle (bool): Shuffle unfixed users if True.
        fixed_perc (float): Percentage of fixed users.
    """
    np.random.seed(1)  # Ensure reproducibility
    conf = 0.95  # Confidence level for computations
    accuracy = 1000  # Number of points for accuracy in computations
    fixed_users = int(fixed_perc * N)  # Number of fixed users for simulation

    # Initialize users based on the type of influence specified by `inf`
    match inf:
        case "00":
            users = [ModeratedUser(M, alpha, moderator) for _ in range(N)]
        case "01":
            users = [ModeratedUser(M, alpha, moderator, c=True) for _ in range(N)]
        case "10":
            users = [SimpleAvgUser(M, alpha) for _ in range(N)]
        case "11":
            users = [SimpleAvgUser(M, alpha, c=True) for _ in range(N)]
        case "20":
            users = [SineUser(M, alpha) for _ in range(N)]

    # Compute statistical properties of the user population
    true_avg = stat.mean([x.rating for x in users])  # True average rating
    a_mean = stat.mean([x.Lambda(0) for x in users])  # Average influence factor
    cov = np.cov([x.rating for x in users], [x.Lambda(0) for x in users])[0, 1]  # Covariance

    # Shuffle unfixed users if the flag is set
    if flag_shuffle:
        unfixed = users[fixed_users:]
        random.shuffle(unfixed)
        for i in range(fixed_users, N):
            users[i] = unfixed[i - fixed_users]

    # Perform the experiment if flag_exp is set
    if flag_exp:
        mean = [users[0].rating] * N  # Initialize mean ratings
        for i in range(1, N):
            if i < fixed_users:
                # Simulate review bombing for fixed users
                r = max(0, 0.2 + np.random.normal(0, 0.2))
            else:
                r = users[i].rate(mean[i - 1])  # Regular user rating
            mean[i] = (mean[i - 1] * i + r) / (i + 1)  # Update mean rating

        # Export experimental results to a file
        with open('../simulations/%s/%d_experiment_%d_%.2f_%.3f_%.2f.txt' % (inf, sim_id, N, M, alpha, moderator), 'w') as f:
            f.write(f"{true_avg:.4f}\t{a_mean:.4f}\t{cov:.4f}\t{mean[N - 1]:.4f}\n")
            for i in range(N):
                user = users[i]
                f.write(f"{user.rating:.4f}\t{user.Lambda(0):.4f}\t{mean[i]:.4f}\n") 

    # Compute the social influence curve (SIC) if flag_sic is set
    if flag_sic:
        xs = np.linspace(0, 1, accuracy)  # Generate x values for SIC computation
        if not empirical_sic:
            ys, upper, lower, equilibria = users[0].sic_computation(xs, users, N, accuracy)
        else:
            u = User(0.5, 1)  # Default user for empirical SIC computation
            ys, upper, lower, equilibria = u.sic_computation(xs, users, N, accuracy, conf)

        # Export SIC results to a file
        with open('../simulations/%s/%d_sic_%d_%.2f_%.3f_%.2f.txt' % (inf, sim_id, N, M, alpha, moderator), 'w') as f:
            for i, j, w, z in zip(xs, ys, upper, lower):
                f.write(f"{i:.4f}\t{j:.4f}\t{w:.4f}\t{z:.4f}\n")
            for eq in equilibria:
                f.write(f"{eq:.4f}\n")
