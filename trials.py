import random as rand
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def sigmoid(b, x):
    return 1 / (1 + np.exp(-b*x))

# function to approximate when y=f(x) intersects with y=x
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

    def __init__(self, i, r, m = 0.5, s = 0.6):
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

# model parameters
np.random.seed(1)
N = 10000        # users
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
no_exp = 500
eq = [0]*no_exp
for e in range(no_exp):
    rand.shuffle(users)
    mean = users[0].rating
    User.inf_sum = users[0].rating
    for i in range(1,N):
        users[i].rate(mean)
        mean = User.inf_sum/(i+1)
    eq[e] = mean

accuracy = 100
m = [i/accuracy for i in range(accuracy)]
y = [0]*accuracy
for i in range(accuracy):
    User.inf_sum = 0
    for j in range(N):
        users[j].rate(m[i])
    y[i] = User.inf_sum/N
        
sigm_eq = apx(m, y)

distr = {str(x):[0,0] for x in sigm_eq}
for i in eq:
    (_, b, a) = min([((i - j)**2,j,i) for j in sigm_eq], key = lambda elem:elem[0])
    distr[str(b)][0] += 1
    distr[str(b)][1] += a
    

plt.figure()
plt.bar(distr.keys(), list(map(lambda x: x[0]/no_exp, distr.values())), 0.4)
plt.xlabel("Equilibria")
plt.ylabel("Percentage of experiments")
plt.title("Permutation Experiment with %d trials and mean rating %.2f" %(no_exp, true_mean))
plt.show()

c = (x for x in ['navy', 'orange', 'orchid'])
plt.figure()
plt.plot(eq, 'o', color='black', markersize=5)
for e in sigm_eq:
    color = next(c)
    plt.plot([e]*no_exp, label = "expected equilibrium", color = color)
    if (distr[str(e)][0]): plt.plot([distr[str(e)][1]/distr[str(e)][0]]*no_exp, linestyle = "dashed", color = color)
plt.yticks(sigm_eq + [distr[str(e)][1]/distr[str(e)][0] for e in sigm_eq if distr[str(e)][0]])
plt.legend()
plt.show()


