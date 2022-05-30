import numpy as np
import matplotlib.pyplot as plt

import lp_solver
from IterativePlayer import one_hot_argmax

"""
    A specialized solver for an mx2 matrix game.
"""

seed = np.random.choice(10000)
print(f"Seed: {seed}")
np.random.seed(seed)

game = np.random.normal(size=(12,2))
# game = np.array([[0,-1],[1,0],[-1,1]])
# game = np.array([[1,-1],[-1,1]])

# Initialize player strategies
alpha = 0.5 
x = np.zeros(game.shape[0])
x[0] = 1

p2_probs, res = lp_solver.solve_game_half(game)
value = res["res"]["fun"]

worst_case_p1 = []
alphas = [alpha]
for t in range(1,1000):
    step_size = 1/((t+1) * np.log(t+1))
    assert 0 <= step_size  <= 1, f"step size must be between 0 and 1, was {step_size} at time {t}"
    
    br = one_hot_argmax(alpha * game[:,0] + (1-alpha) * game[:,1])
    x = (1-step_size)*x + step_size*br
    
    p2_payoffs = - x @ game
    if p2_payoffs[0] < p2_payoffs[1]:
        alpha = (1-step_size)*alpha
    else:
        alpha = (1-step_size)*alpha + step_size 
    
    alphas.append(alpha)
    worst_case_p1.append(-np.max(p2_payoffs))
    

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(5,10))

ax1.plot(alphas, label="alpha")
ax1.axhline(p2_probs[0], c="C1", label="Nash prob of action 0", ls="--")
ax1.set_ylabel("Alpha")
ax1.set_xlabel("Timestep")
ax1.set_ylim(0,1)
# ax1.set_title("Player 2 probability")
ax1.legend()

ax2.plot(worst_case_p1, label="worst case payoff", c="gray")
ax2.axhline(value, c="black", label="value of game", ls="--")
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Worst-case payoff (player 1)")
ax2.legend()

