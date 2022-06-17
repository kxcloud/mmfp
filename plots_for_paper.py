from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import ternary

import IterativePlayer

seed = np.random.choice(10000)
print(f"Seed: {seed}")
np.random.seed(seed)

games = IterativePlayer.games

#%% Game-specific comparisons

action_names_dict = {
    "RPS" : ["Rock", "Paper", "Scissors"],
    "RPS Abstain" : ["Rock", "Paper", "Scissors", "Abstain"],
    "Biased RPS" : ["Rock", "Paper", "Scissors"],
    "weakRPS" : ["Rock", "Paper", "Scissors"],
    "Matching Pennies Abstain" : ["Heads", "Tails", "Abstain"]
}


def get_exploitability_streaks(play):
    on_streak = np.zeros(play.t)
    for t in range(1, play.t):
        same_action = np.array_equal(play.p1_response[t,:], play.p1_response[t-1], equal_nan=True)
        increased_exploitability = play.worst_case_payoff[t,0] < play.worst_case_payoff[t-1,0]
        if same_action and increased_exploitability:
            on_streak[t] = 1
    return on_streak


def plot_alg_behavior(plays_by_alg_dict, game_name):
    num_runs = len(plays_by_alg_dict)
    fig, axes = plt.subplots(
        nrows=num_runs+1, ncols=1, sharex=True, figsize=(7,10)
    )
    
    for idx, (label, play) in enumerate(plays_by_alg_dict.items()):
        IterativePlayer.plot_single_player(play.p1_response, play.p1_empirical, ax=axes[idx], title=label)
        
        streaks = get_exploitability_streaks(play)
        axes[idx].plot(streaks, c="red", alpha=0.5, ls=":")
        axes[-1].plot(play.worst_case_payoff[:,0], lw=2, label=label, c=f"C{5+idx}")
    
    fig.suptitle(f"Algorithm behavior on {game_name}")
    axes[-1].set_ylabel("Worst-case payoff")
    axes[-1].legend()
    axes[-1].set_title("Performance comparison")
    
    axes[0].set_ylabel("Probability")
    axes[-1].set_xlabel("Timestep")
    return axes

def plot_on_simplex(plays_by_alg_dict, num_best_responses_to_plot, action_names, game_name):
    fig, axes = plt.subplots(
        ncols=len(plays_by_alg_dict), figsize=(13,4), sharey=True, sharex=True
    )
    fig.suptitle(f"{game_name}")
    
    for idx, (label, play) in enumerate(plays_by_alg_dict.items()):
        num_to_plot = np.sum(play.total_compute[:,0] <= num_best_responses_to_plot)
        
        ax = axes[idx]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        
        response_plot = play.p1_response[:num_to_plot]
        empirical_plot = play.p1_empirical[:num_to_plot]
        
        tax = ternary.TernaryAxesSubplot(ax=ax)
        tax.plot(empirical_plot, c="black", lw=1)
        
        strat_colors = [f"C{idx}" for idx in np.where(response_plot==1)[1]]
        strat_sizes = np.linspace(40, 1, num_to_plot)
        tax.scatter(empirical_plot, zorder=9, color=strat_colors, s=strat_sizes)
        tax.scatter([play.p1_probs_nash], marker="*", c="black", s=60, zorder=10)
        tax.set_title(f"{label} ({num_to_plot} steps)")
        
    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color="w", 
            markerfacecolor=f"C{idx}", markersize=8, label=action_name
        )
        for idx, action_name in enumerate(action_names)
    ]
    axes[-1].legend(handles=legend_elements)

def qplot(game_name, t_max, noise=None):
    game = games[game_name]
        
    if game_name in action_names_dict:
        action_names = action_names_dict[game_name]
        
    if noise is not None:
        game = game + np.random.normal(size=game.shape, scale=noise)
        game_name += " (noisy)"
        
    initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
    initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])


    plays_by_alg_dict = {
        f"AFP({k})" : 
        IterativePlayer.run_afp_general(game, t_max, initial_strategy_p1, initial_strategy_p2, steps_to_anticipate=k)
        for k in range(4)
    }
    # for k in range(5):
    
    # play_fp = IterativePlayer.run_fp(game, t_max, initial_strategy_p1, initial_strategy_p2)
    # play_afp = IterativePlayer.run_afp_general(game, t_max, initial_strategy_p1, initial_strategy_p2, steps_to_anticipate=3)
    
    # plays_by_alg_dict = {
    #     "FP" : play_fp,
    #     "AFP" : play_afp        
    # }
    
    print()
    print()
    print(game.round(3))
    plot_alg_behavior(plays_by_alg_dict, game_name=game_name)
    plt.show()
    if game.shape[0] == 3:
        plot_on_simplex(plays_by_alg_dict, t_max, action_names, game_name=game_name)    
    plt.show()

t_max = 50

print()
print()
print("------------------")
for game_name, action_names in action_names_dict.items():
    qplot(game_name, t_max, noise=None)

#%% Average performance over fixed size

game_shape = (30,30)

alg_names = []
alg_list = []
alg_br_per_timestep = []

for k in range(4):
    alg = partial(
        IterativePlayer.run_afp_general, 
        initial_strategy_p1 = IterativePlayer.one_hot(0, game_shape[0]),
        initial_strategy_p2 = IterativePlayer.one_hot(0, game_shape[1]),
        steps_to_anticipate=k
    )
    alg_names.append(f"FP({k})")
    alg_list.append(alg)
    alg_br_per_timestep.append(k+1)

num_game_samples = 20
t_max = 100

avg_worst_case_payoffs = {    
    name: np.zeros((t_max, 2)) for name in alg_names
}

pct_of_time_better_than_first_alg = {
    name: np.zeros(t_max) for name in alg_names[1:]
}


for _ in range(num_game_samples):
    game = np.random.normal(size=game_shape)
    
    for k, (alg_name, alg, br_per_step) in enumerate(zip(alg_names, alg_list, alg_br_per_timestep)):
        play = alg(game, t_max)
        performance = play.worst_case_payoff[::br_per_step,:]
        
        avg_worst_case_payoffs[alg_name] += performance
        
        if k == 0:
            first_alg_performance = performance
        
        if k > 0:
            pct_of_time_better_than_first_alg[alg_name] += first_alg_performance <= performance

for name in alg_names:
    avg_worst_case_payoffs[name] /= num_game_samples

for name in alg_names[1:]:
    pct_of_time_better_than_first_alg /= num_game_samples        
        
# x_vals = list(range(2,2*t_max,2))

# fig, ax = plt.subplots()
# ax.plot(x_vals, avg_worst_case_payoff_fp[1:,0],label="FP")
# ax.plot(x_vals, avg_worst_case_payoff_afp[1:,0], label="AFP")
# ax.set_ylabel("Mean worst-case payoff")
# ax.set_xlabel("Best responses calculated")
# ax.set_title("Performance on random 30x30 matrices")
# ax.legend()

# fig, ax = plt.subplots()
# ax.plot(x_vals,pct_of_time_afp_better_fp[1:])
# ax.set_ylabel("Proportion")
# ax.set_xlabel("Best responses calculated")
# ax.set_title("Proportion of timesteps where AFP is better than FP\n(random 30x30 matrices)")
# plt.show()

#%% Performance by size

num_reps = 1_000
t_max = 50

n_vals = range(2,20)

results = []
for n in n_vals:
    print(f"n={n}...")
    avg_worst_case_payoff_fp = 0 
    avg_worst_case_payoff_afp = 0 
    
    for _ in range(num_reps):
        game = np.random.normal(size=(n,n))
        initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
        initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])
        
        play_fp = IterativePlayer.run_fp(game, 2*t_max, initial_strategy_p1, initial_strategy_p2)
        avg_worst_case_payoff_fp += play_fp.worst_case_payoff[-1, 0]
        
        play_afp = IterativePlayer.run_afp(game, t_max, initial_strategy_p1, initial_strategy_p2)
        avg_worst_case_payoff_afp += play_afp.worst_case_payoff[-1, 0]
    avg_worst_case_payoff_fp /= num_reps
    avg_worst_case_payoff_afp /= num_reps
    results.append((n, avg_worst_case_payoff_fp, avg_worst_case_payoff_afp))
    
results = np.array(results)

fig, ax = plt.subplots()
ax.plot(results[:,0], results[:,1],label="FP")
ax.plot(results[:,0], results[:,2],label="AFP")
ax.set_ylabel("Mean worst-case payoff")
ax.set_xlabel("Matrix width and height")
ax.set_xticks(n_vals)
ax.set_title(f"Performance on random matrices after {2*t_max} responses calculated")
ax.legend()
