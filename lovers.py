import random as rand
import numpy as np
import matplotlib.pyplot as plt

# model parameters
N = 1000    # users
lovers_perc = 0.01
lovers = int(N*lovers_perc)
lovers_rating = 4
haters_perc = 1 - lovers_perc
haters = N - lovers
haters_rating = 2

M = 11
influence = [i/(M-1) for i in range(0, M)]
true_rating = lovers_perc*lovers_rating + haters_perc*haters_rating
avg_last = []

# types of arrivals
rand.seed(18)
random = [lovers_rating]*lovers + [haters_rating]*haters
rand.shuffle(random)

random_switch = random.copy()
random_switch[0] = 6 - random_switch[0]

random_switch1 = random.copy()
random_switch1[1] = 6 - random_switch1[1]


# first = 0.1
# help = [lovers_rating]*int((1-first)*lovers) + [haters_rating]*haters
# rand.shuffle(help)
# lover_heavy = [lovers_rating]*int(first*lovers) + help

# help = [lovers_rating]*lovers + [haters_rating]*int((1-first)*haters)
# rand.shuffle(help)
# hater_heavy = [haters_rating]*int(first*haters) + help


# experiment
ARR = ["random", "random_switch"]
for arr in ARR:
    plt.figure(figsize=(10, 5))
    for m in range(M):
        inf = influence[m]
        if arr == "random": arrivals = random
        elif arr == "lover_heavy": arrivals = lover_heavy
        elif arr == "random_switch": arrivals = random_switch
        elif arr == "random_switch1": arrivals = random_switch1
        else: arrivals = hater_heavy
        
        ratings = [arrivals[0]]
        avg = [arrivals[0]]

        for i in range(1,N):
            rating = inf*np.mean(ratings) + (1-inf)*arrivals[i]
            ratings.append(rating)
            avg.append(np.mean(ratings))
            
        plt.plot(avg, label="influence = " + str(inf)[0:4])
        avg_last.append(np.mean(ratings))

    plt.plot([true_rating]*N, label = "true rating")
    plt.title("Lovers rating: " + str(lovers_rating) + " Haters rating: " + str(haters_rating) + " Percentage of lovers: " + str(lovers_perc))
    plt.xlabel("Users")
    plt.ylabel("Average rating")
    plt.legend()
    plt.savefig("plots/lovers" + str(lovers_perc) + "_" + arr + ".png", dpi=300)
    plt.show()

# plt.figure()
# plt.plot(influence, avg_last[:M], label = "random")
# plt.plot(influence, avg_last[M:], label = "switch")
# plt.legend()
# plt.xlabel("Influence parameter")
# plt.ylabel("Final Average rating after 1000 users")
# plt.savefig("plots/last_rating" + str(lovers_perc) + "_" + arr + ".png")
# plt.show()

# diff = [(avg_last[i]-avg_last[M+i]) for i in range(M)]
# plt.figure()
# plt.plot(influence, diff)
# plt.xlabel("Influence parameter")
# plt.title("Difference in final average rating after 1st user is switched")
# plt.savefig("plots/diff" + str(lovers_perc) + "_" + str(lovers_rating) + "_" + str(haters_rating) + ".png")
# plt.show()

plt.figure()
a = [i/100 for i in range(500)]
for inf in influence:
    if inf == 1: plt.plot(a,a, label="Influence = 1", linestyle='dashed')
    else: plt.plot(a, [inf*i + (1-inf)*(true_rating) for i in a], label="Influence = " + str(inf))

plt.vlines(x=true_rating, ymin=0, ymax=true_rating, linestyles='dotted', colors='black', label="x = true_rating")
plt.title("Social influence")
plt.xlabel("Mean rating")
plt.ylabel("Expected rating")
plt.legend()
plt.savefig("plots/social_influence" + str(lovers_perc) + "_" + str(lovers_rating) + "_" + str(haters_rating) + ".png")
plt.show()