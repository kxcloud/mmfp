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
    
    anticipation_counter = 0
    last_locked_in_idx = 0
    num_locked_in = 1
    
    for t in range(1, t_max):
        sample_size = num_locked_in + anticipation_counter * (anticipation_counter != anticipation_level)
        idx_to_respond_to = last_locked_in_idx if anticipation_counter == 0 else t-1
        idx_to_avg_with = last_locked_in_idx if anticipation_counter == anticipation_level else t-1
                    
        for player_idx in [0,1]:
            opponent_idx = 1-player_idx
            payoffs = payoff_matrix[player_idx] @ empirical[opponent_idx][idx_to_respond_to]
            responses[player_idx][t] = one_hot_argmax(payoffs)
            
            empirical[player_idx][t] = (
                (sample_size*empirical[player_idx][idx_to_avg_with] 
                 + responses[player_idx][t])/(sample_size+1)
            )
        
        if anticipation_counter == anticipation_level:
            anticipation_counter = 0
            last_locked_in_idx = t
            num_locked_in += 1
        else:
            anticipation_counter += 1

    assert np.min(np.abs(empirical[0].sum(axis=1) - 1) < 1e-6), "Probs must sum to 1"
    return responses, empirical

def plot(responses, empirical, ax, player_idx=0, subsample=1):
    ax.plot(empirical[player_idx][::subsample])
    
    for strategy_idx in range(game.shape[player_idx]):
        color = f"C{strategy_idx}"
        
        action_inds = responses[player_idx][::subsample, strategy_idx].copy()
        action_inds[action_inds == 0] = np.nan
        
        ax.scatter(range(t_max//subsample), action_inds, lw=0.1, c=color, s=15)    


#%% Subsample to old level 
game_name = "RPS"
t_max = 100
game = games.game_dict[game_name]

# anticipation_level = 1

# responses, empirical = run_afp(game, t_max, anticipation_level)
# fig, ax = plt.subplots()
# plot(responses, empirical, ax)

# fig, ax = plt.subplots()
# plot(responses, empirical, ax, subsample=anticipation_level+1)

#%% Plot many

anticipation_levels = range(2)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharex=True)
fig.suptitle(game_name)
_, perf_ax = plt.subplots(figsize=(10,10))

axes_flat = axes.reshape(-1)
for idx, anticipation_level in enumerate(anticipation_levels):
    ax = axes_flat[idx]
    responses, empirical = run_afp(game, t_max, anticipation_level)
    ax.plot(empirical[0])
    
    for strategy_idx in range(game.shape[0]):
        color = f"C{strategy_idx}"
        
        action_inds = responses[0][:, strategy_idx].copy()
        action_inds[action_inds == 0] = np.nan
        
        ax.scatter(range(t_max), action_inds, lw=0.1, c=color, s=15)    
    
    label = f"AFP({anticipation_level})"
    ax.set_title(label)
    
    worst_case = get_worst_case_payoffs(game, empirical)
    axes_flat[-1].plot(worst_case[0], label=label)
    perf_ax.plot(worst_case[0], label=label)

for ax in [axes_flat[-1], perf_ax]:
    ax.legend()
    ax.set_xlabel("Timesteps (=best responses)")
    ax.set_title(f"Worst-case performance on {game_name}")
    
    
#%% Compare average performance

# t_max = 100
# num_random_matrices = 1000
# game_sizes = range(2,30)
# anticipation_levels = range(5)

# worst_case_payoffs = np.zeros((len(game_sizes), len(anticipation_levels), num_random_matrices, t_max))

# for game_size_idx, game_size in enumerate(game_sizes):
#     for matrix_idx in range(num_random_matrices):
#         game = np.random.normal(size=(game_size, game_size))
        
#         for anticipation_idx, anticipation_level in enumerate(anticipation_levels):
#             responses, empirical = run_afp(game, t_max, anticipation_level)
#             worst_case = get_worst_case_payoffs(game, empirical)[0]
#             worst_case_payoffs[game_size_idx, anticipation_idx, matrix_idx] += worst_case

# #%% Plot data
# fig, ax = plt.subplots()
# for idx, anticipation_level in enumerate(anticipation_levels):
#     performance = worst_case_payoffs[:,anticipation_level,:,-1]
#     mean_perf = np.mean(worst_case_payoffs[:,anticipation_level,:,-1], axis=1)
#     std_perf = np.std(worst_case_payoffs[:,anticipation_level,:,-1], axis=1)
#     ci_width = 1.96*std_perf / np.sqrt(num_random_matrices)

#     ax.plot(mean_perf, label=f"AFP({anticipation_level})")    
#     ax.plot(mean_perf - ci_width, c=f"C{idx}", ls="--", alpha=0.3)
#     ax.plot(mean_perf + ci_width, c=f"C{idx}", ls="--", alpha=0.3)
    
# ax.legend()
# ax.set_title(f"Average performance of AFP(k) on random matrices\n at t={t_max} best responses calculated")
# ax.set_xlabel("Matrix size")
# ax.set_ylabel("Worst case payoffs")