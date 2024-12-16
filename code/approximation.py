import numpy as np
import scipy

# Computes the t-critical value for a given confidence level and sample size
def t_critical_value(confidence_level, sample_size):
    df = sample_size - 1  # Degrees of freedom
    alpha = 1 - confidence_level  # Significance level
    t_critical = scipy.stats.t.ppf(1 - alpha / 2, df)
    return t_critical

# Computes the sample standard deviation for a given dataset
def sample_standard_deviation(data, N, mean):
    variance = sum((x - mean) ** 2 for x in data) / (N - 1)
    return np.sqrt(variance)

# Finds points of intersection between two curves, m and y
def apx(m, y):
    ans = []  # Stores intersection points
    flag = y[0] < m[0]  # Initial comparison flag
    for i in range(len(m)):
        if flag != (y[i] < m[i]):
            flag = not flag
            ans.append(m[i])  # Add intersection point
    return ans

# Approximates confidence intervals for curve intersections
def apx_conf(m, u, l):
    ans = []  # Stores resulting points
    prev = -1  # Track previous point for thresholding
    curr = []  # Temporary list for interval averaging
    for i in range(len(m)):
        # Check if m[i] lies within the confidence interval [l[i], u[i]]
        if l[i] <= m[i] and u[i] >= m[i]:
            if m[i] - prev > 0.05:  # Threshold check
                if curr:
                    ans.append(np.mean(curr))  # Add averaged points
                    curr = []
            curr.append(m[i])
            prev = m[i]
    if curr:
        ans.append(np.mean(curr))
    return ans

# Computes the cumulative distribution function (CDF) for ratings
def cdf(ratings, xs, N):
    counter = 0  # Tracks the number of ratings less than or equal to x
    for x in xs:
        while counter < N and ratings[counter] <= x:
            counter += 1
        yield counter
