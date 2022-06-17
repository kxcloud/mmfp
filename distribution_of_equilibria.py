import numpy as np
import matplotlib.pyplot as plt

import lp_solver

def sample_num_nash(game_size, num_matrices):
    '''
    Sample num_matrices (game_size, game_size) antisymmetric matrices and 
    return an array with the number of strategies in a Nash support for each
    matrix. 
    ''' 
    nash_support_size = np.zeros(num_matrices)
    for idx in range(num_matrices):
        A = np.random.normal(size=(game_size,game_size))
        game = A - A.T
        probs, _ = lp_solver.solve_game_half(game)
        nash_support_size[idx] = np.sum(probs.round(5) > 0)
    return nash_support_size

# sample_size = 500
# for game_size in [5, 10, 30, 100]:
#     nash_support_size = sample_num_nash(game_size, sample_size)
    
#     plt.hist(nash_support_size, bins=range(1,game_size+2))
#     plt.title(f"Game size: {game_size}, n={sample_size}")
#     plt.xlabel("Number of actions in Nash support")
#     plt.ylabel("Frequency")
#     plt.show()
    
sample_size = 200
game_sizes = range(2,35)
results = np.zeros((len(game_sizes), 3)) # mean, 5th pct, 95thpct
for idx, game_size in enumerate(game_sizes):
    res = sample_num_nash(game_size, sample_size)
    results[idx] = (res.mean(), *np.quantile(res, [0.05,0.95]))
    
fig, ax = plt.subplots()

ax.set_title(f"Nash support by random matrix size (n={sample_size} per matrix size)")
ax.set_ylabel("Number of actions in Nash support")
ax.set_xlabel("Matrix size")
ax.plot(game_sizes, results[:,0], label="Mean")
ax.plot(game_sizes, results[:,1], label="5th, 95th pctiles", c="C0", ls="--")
ax.plot(game_sizes, results[:,2], c="C0", ls="--")
ax.legend()