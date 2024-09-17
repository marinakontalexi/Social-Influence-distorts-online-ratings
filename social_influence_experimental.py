import random as rand
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def sigmoid(b, x):
    return 1 / (1 + np.exp(-b*x))

# function to approximate when m intersects with y
def apx(m, y):
    ans = []
    flag = y[0] <= m[0]
    for i in range(len(m)):
        if (flag != (y[i] <= m[i])):
            flag = not flag
            ans.append(m[i])
    return ans

class User:
    inf_sum = 0
    b = 0.5
    inf = 0

    def __init__(self, i, r, m = 0.6, s = 0.6):
        self.rating = min(1, max(0, np.random.normal(loc = m, scale = s)))
        # self.rating = np.random.random()
        self.influence_type = i
        self.rating_type = r        

    def rate(self, mean, a = 0.5, b = 0, c = 0):
        if (self.influence_type == "uniform"):
            self.inf = a
        elif (self.influence_type == "random"):
            self.inf = np.random.random()
        elif (self.influence_type == "var"):
            self.inf = (abs(mean - self.rating)/max(mean, 1-mean))**2
        elif (self.influence_type == "sigmoid"):
            self.inf = sigmoid(User.b, abs(self.rating - mean))

        if (self.rating_type == "linear"):
            User.inf_sum += self.inf*mean + (1-self.inf)*self.rating
        elif (self.rating_type == "sigmoid"):
            User.inf_sum += self.inf*sigmoid(b, mean - c) + (1-self.inf)*self.rating
        return

# model parameters
np.random.seed(1)
N = 10000       # users
one_perc = 0    # flag to order first 1% of users ->  
                # 1: ascending   -1: descending   0: no ordering

# init
users = [0]*N
for i in range(N):
    users[i] = User("var", "linear")

if (one_perc):
    u = sorted(users[:int(N/100)], key = lambda elem: one_perc*elem.rating)
    for i in range(int(N/100)):
        users[i] = u[i]

true_mean = stat.mean([x.rating for x in users])
s = stat.stdev([x.rating for x in users])
# print("True rating:", true_mean, "\nStandard deviation:", s)


# experiment
mean = np.zeros(N)
mean[0] = users[0].rating
User.inf_sum = mean[0]
for i in range(1,N):
    users[i].rate(mean[i-1])
    mean[i] = User.inf_sum/(i+1)
print("Mean influence:", stat.mean([x.inf for x in users]))

# influence curve calc
accuracy = 100
m = [i/accuracy for i in range(accuracy)]
y = [0]*accuracy
for i in range(accuracy):
    User.inf_sum = 0
    for j in range(N):
        users[j].rate(m[i])
    y[i] = User.inf_sum/N
        
sigm_eq = apx(m, y)

# plot mean rating 
plt.figure()
plt.plot(mean)
plt.plot([true_mean]*N, label = "true rating")
for eq in sigm_eq:
    plt.plot([eq]*N, label = "possible equilibrium", linestyle = "dashed")
plt.yticks([min(mean) + i/10 for i in range(int(10*(max(mean) - min(mean))) + 1)] + sigm_eq + [true_mean])
plt.xlabel("Users")
plt.ylabel("Average rating")
plt.legend()
# plt.savefig("plots/mean_var.png")
plt.show()

# plot social influence curve
plt.figure()
plt.plot(m, y)
plt.plot(m, m, linestyle='dashed', label = "y = x")
for eq in sigm_eq:
    plt.vlines(x=eq, ymin=0, ymax=eq, linestyles='dotted', colors='black', label="x = " + str(eq))
plt.title("Social influence curve")
plt.xlabel("Mean rating")
plt.ylabel("Expected rating")
plt.legend()
# plt.savefig("plots/curve_var.png")
plt.show()
