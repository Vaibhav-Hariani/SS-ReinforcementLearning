import numpy as np
import random
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    ## T time steps, N arms, mu & sig for random
    T = 1000
    N = 10
    ep = 0.1
    mu = 50
    ##Each choice is a completely unique distribution
    ##They are all roughly similar, but the value of Q_T should converge to argmax(Q_prime)
    sig_outer = 10
    sig_inner = 5
    Q_prime = np.random.normal(mu,sig_outer, N)
    Q_t = np.zeros(N)
    N_t = np.zeros(N)
    avg = 0
    for t in range(1,T):
        if ep > np.random.random():
            choice = np.random.choice(N)
        else:
            choice = np.argmax(Q_t)
        val = np.random.normal(Q_prime[choice], sig_inner)
        N_t[choice] += 1
        Q_t[choice] = Q_t[choice] + (val - Q_t[choice])/N_t[choice]
        avg = ((avg * t-1) + val) / t

    print(f"Target choice is {np.argmax(Q_prime)} ")
    print(f"Learned choice is {np.argmax(Q_t)}")
    print(f"Average reward is {avg}")
    for t in range(N):
        print(f"Chose {t} {N_t[t]} times: Estimated at {Q_t[t]}, Actual {Q_prime[t]}")
