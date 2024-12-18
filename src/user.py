import numpy as np
from approximation import *  # Import custom approximation functions
import statistics as stat

# Concave asymptotic function
# Models a decreasing return with respect to input x
def concave_asymptotic(x):
    return 1 - 1 / (1 + 0.01 * x)

# Constraint function
# Returns the maximum of m and 1-m
def C(m):
    return max(m, 1 - m)

# Base class representing a user
class User:
    a = 3
    def __init__(self, m, alpha, normalize=True):
        # User's rating initialized using a Beta distribution
        self.rating = np.random.beta(alpha, alpha / m - alpha)
        self.normalize = normalize

    def Lambda(self, *args, **kwargs):
        # Placeholder method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the 'Lambda' method")

    def rate(self, x):
        # Computes the rating influenced by the 'information' (inf) weight
        inf = np.random.beta(User.a, User.a / self.Lambda(x) - User.a)
        return inf * x + (1 - inf) * self.rating

    def sic_computation(self, xs, users, N, accuracy, conf=0.95):
        # Computes statistics for confidence intervals
        upper = np.array(accuracy)
        lower = np.array(accuracy)
        ys = np.zeros(accuracy)
        for i, x in enumerate(xs):
            # Aggregate ratings from all users
            R = [u.rate(x) for u in users]
            ys[i] = np.mean(R)
            s = sample_standard_deviation(R, N, ys[i])
            t = t_critical_value(conf, N)
            upper[i] = ys[i] + t * s / np.sqrt(N)
            lower[i] = ys[i] - t * s / np.sqrt(N)
        return ys, upper, lower, apx_conf(xs, upper, lower)

# A user moderated by a specific influence (e.g., social norms)
class ModeratedUser(User):
    def __init__(self, m, alpha, moderator, c=False, normalize=True):
        super().__init__(m, alpha, normalize)
        self.mod = moderator
        self.contrasting = c

    def Lambda(self, x=0):
        # Computes influence based on the moderator's rating
        C_val = 1 if not self.normalize else C(self.mod)
        if self.contrasting:
            return 1 - abs(self.mod - self.rating) / C_val
        return abs(self.mod - self.rating) / C_val

    def sic_computation(self, xs, users, N, accuracy, conf=0.95):
        # Computes aggregated influence values for users
        ys = np.zeros(accuracy)
        Er = stat.mean([x.rating for x in users])
        a_mean = stat.mean([x.Lambda() for x in users])
        cov = np.cov([x.rating for x in users], [x.Lambda() for x in users])[0, 1]
        for i in accuracy:
            ys[i] = a_mean * xs[i] + (1 - a_mean) * Er - cov
        return ys, np.zeros(accuracy), np.zeros(accuracy), apx(xs, ys)

# A user that averages values simply
class SimpleAvgUser(User):  
    def __init__(self, m, alpha, c=False, normalize=True):
        super().__init__(m, alpha, normalize)
        self.contrasting = c

    def Lambda(self, x):
        # Computes influence based on the current value x
        C_val = 1 if not self.normalize else C(x)
        if self.contrasting:
            return 1 - abs(x - self.rating) / C_val
        return abs(x - self.rating) / C_val

    def sic_computation(self, xs, users, N, accuracy):
        # Computes aggregated influence statistics for the user
        counter = 0
        Er = 0
        Er2 = 0
        ratings = sorted([x.rating for x in users])
        ys = np.zeros(accuracy)
        p = np.zeros(accuracy)

        for i, x in enumerate(xs):
            # Forward pass to calculate intermediate statistics
            while counter < N and ratings[counter] <= x:
                counter += 1
                Er = (Er * (counter - 1) + ratings[counter - 1]) / counter
                Er2 = (Er2 * (counter - 1) + ratings[counter - 1]**2) / counter
            p[i] = counter / N
            ys[i] = p[i] * (x**2 - 2 * x * Er + Er2)

        # Backward pass for the remaining calculations
        Er = 0
        Er2 = 0
        counter = 0
        for i in range(accuracy - 1, -1, -1):
            x = xs[i]
            while counter < N and ratings[N - 1 - counter] > x:
                counter += 1
                Er = (Er * (counter - 1) + ratings[N - counter]) / counter
                Er2 = (Er2 * (counter - 1) + ratings[N - counter]**2) / counter
            ys[i] -= (1 - p[i]) * (x**2 - 2 * x * Er + Er2)

        # Normalize the results if necessary
        for i in range(accuracy):
            C_val = 1 if not self.normalize else C(xs[i])
            if self.contrasting:
                ys[i] = xs[i] - ys[i] / C_val
            else:
                ys[i] = ys[i] / C_val + Er
        return ys, np.zeros(accuracy), np.zeros(accuracy), apx(xs, ys)

# A user that computes influence using a sine function
class SineUser(User):
    def Lambda(self, x):
        # Computes influence based on the sine of the distance to the user's rating
        d = x - self.rating
        if abs(d) >= 0.5:
            return np.normal(0, 0.01)  # Small random influence if the difference is large
        return 0.5 * np.sin(d * np.pi / 0.5) / d
