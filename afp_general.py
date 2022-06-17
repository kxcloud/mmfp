import matplotlib.pyplot as plt
import numpy as np

import games

import matplotlib.style as style
style.use('tableau-colorblind10')

def one_hot(index, length):
    array = np.zeros(length)
    array[index] = 1
    return array

def one_hot_argmax(array):
    one_hot = np.zeros_like(array)
    favor_lower_indices = np.linspace(0,-1e-6, len(array))
    one_hot[np.argmax(array+favor_lower_indices)] = 1
    
    return one_hot

def get_worst_case_payoffs(game, empirical):
    return np.array([
        np.min(empirical[0] @ game, axis=1),
        np.min(empirical[1] @ -game.T, axis=1)
    ])
             

def run_afp(game, t_max, anticipation_level, initial_strategies=[0,0]):
    """
    anticipation_level=0 -> FP
    anticipation_level=1 -> regular AFP
    """
    # Initialization
    payoff_matrix = [game, -game.T]
    responses = [np.zeros((t_max, game.shape[0])), np.zeros((t_max, game.shape[1]))]
    empirical = [np.zeros((t_max, game.shape[0])), np.zeros((t_max, game.shape[1]))]
    
    for player_idx, initial_strategy in enumerate(initial_strategies):
        responses[player_idx][0] = one_hot(initial_strategy, game.shape[player_idx])
        empirical[player_idx][0] = one_hot(initial_strategy, game.shape[player_idx])
    
    alg_idx = 0 # track the most recent "locked in" empirical strategy
    alg_n = 1 # track the sample size for the recent "locked in" strategy
    anticipation_counter = 0
    
    for t in range(1,t_max):
        sample_size = alg_n + anticipation_counter
        idx = alg_idx + anticipation_counter
        
        for player_idx in [0,1]:
            opponent_idx = 1-player_idx
            payoffs = payoff_matrix[player_idx] @ empirical[opponent_idx][alg_idx]
            responses[player_idx][t] = one_hot_argmax(payoffs)
            empirical[player_idx][t] = (
                (sample_size*empirical[player_idx][idx] 
                 + responses[player_idx][t])/(sample_size+1)
            )
        
        anticipation_counter += 1
        if anticipation_counter > anticipation_level:
            # "Lock in" the current strategy
            anticipation_counter = 0
            alg_idx = t
            alg_n += 1
            
    return responses, empirical

game_name = "Biased RPS"
t_max = 150
game = games.game_dict[game_name]

anticipation_levels = range(5)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11,9), sharex=True)
fig.suptitle(game_name)
_, perf_ax = plt.subplots(figsize=(10,10))

axes_flat = axes.reshape(-1)
for idx, anticipation_level in enumerate(anticipation_levels):
    ax = axes_flat[idx]
    responses, empirical = run_afp(game, t_max, anticipation_level)
    ax.plot(empirical[0])
    label = f"AFP({anticipation_level})"
    ax.set_title(label)
    
    worst_case = get_worst_case_payoffs(game, empirical)
    axes_flat[-1].plot(worst_case[0], label=label)
    perf_ax.plot(worst_case[0], label=label)

for ax in [axes_flat[-1], perf_ax]:
    ax.legend()
    ax.set_xlabel("Timesteps (=best responses)")
    ax.set_title(f"Worst-case performance on {game_name}")
