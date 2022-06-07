import numpy as np
import matplotlib.pyplot as plt
import ternary

import IterativePlayer

seed = np.random.choice(10000)
print(f"Seed: {seed}")
np.random.seed(seed)

games = IterativePlayer.games

#%% Game-specific comparisons

action_names = {
    "RPS" : ["Rock", "Paper", "Scissors"],
    "RPS Abstain" : ["Rock", "Paper", "Scissors", "Abstain"],
    "Biased RPS" : ["Rock", "Paper", "Scissors"],
    "weakRPS" : ["Rock", "Paper", "Scissors"],
    "Matching Pennies Abstain" : ["Heads", "Tails", "Abstain"]
}


def plot_alg_behavior(plays_by_alg_dict):
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(6,6)
    )
    
    for idx, (label, play) in enumerate(plays_by_alg_dict.items()):
        IterativePlayer.plot_single_player(play.p1_response, play.p1_empirical, ax=axes[idx], title=label)
        axes[2].plot(play.worst_case_payoff[:,0], lw=2, label=label, c=f"C{5+idx}")
    
    fig.suptitle(f"Algorithm behavior on {game_name}")
    axes[2].set_ylabel("Worst-case payoff")
    axes[2].legend()
    axes[2].set_title("Performance comparison")
    
    axes[0].set_ylabel("Probability")
    axes[-1].set_xlabel("Timestep")
    return axes

def plot_on_simplex(plays_by_alg_dict, num_best_responses_to_plot, action_names):
    fig, axes = plt.subplots(ncols=2, figsize=(9,4.5), sharey=True, sharex=True)
    fig.suptitle(f"{game_name}")
    
    for idx, (label, play) in enumerate(plays_by_alg_dict.items()):
        num_to_plot = num_best_responses_to_plot
        assert label in ["FP", "AFP"]
        if label == "AFP":
            num_to_plot = num_to_plot // 2
        
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
    axes[1].legend(handles=legend_elements)

t_max = 50

for game_name, action_names in action_names.items():
    game = games[game_name] 
    # game = game + np.random.normal(size=game.shape, scale=0.01)
    initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
    initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])
    
    play_fp = IterativePlayer.run_fp(game, t_max, initial_strategy_p1, initial_strategy_p2)
    play_afp = IterativePlayer.run_afp(game, t_max, initial_strategy_p1, initial_strategy_p2)
    
    plays_by_alg_dict = {
        "FP" : play_fp,
        "AFP" : play_afp        
    }
    
    plot_alg_behavior(plays_by_alg_dict)
    if game.shape[0] == 3:
        plot_on_simplex(plays_by_alg_dict, t_max, action_names)    


#%% Average performance over fixed size

## TODO: record every timestep when FP is better than AFP and plot %s

num_reps = 200
t_max = 500

avg_worst_case_payoff_fp  = np.zeros((t_max, 2))
avg_worst_case_payoff_afp = np.zeros((t_max, 2))

pct_of_time_afp_better_fp = np.zeros(t_max)

for _ in range(num_reps):
    game = np.random.normal(size=(30,30))
    initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
    initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])
    
    play_fp = IterativePlayer.run_fp(game, 2*t_max, initial_strategy_p1, initial_strategy_p2)
    avg_worst_case_payoff_fp += play_fp.worst_case_payoff[::2]
    
    play_afp = IterativePlayer.run_afp(game, t_max, initial_strategy_p1, initial_strategy_p2)
    avg_worst_case_payoff_afp += play_afp.worst_case_payoff
    
    pct_of_time_afp_better_fp += play_fp.worst_case_payoff[::2,0] <= play_afp.worst_case_payoff[:,0]
    
avg_worst_case_payoff_fp /= num_reps
avg_worst_case_payoff_afp /= num_reps
pct_of_time_afp_better_fp /= num_reps

x_vals = list(range(2,2*t_max,2))

fig, ax = plt.subplots()
ax.plot(x_vals, avg_worst_case_payoff_fp[1:,0],label="FP")
ax.plot(x_vals, avg_worst_case_payoff_afp[1:,0], label="AFP")
ax.set_ylabel("Mean worst-case payoff")
ax.set_xlabel("Best responses calculated")
ax.set_title("Performance on random 30x30 matrices")
ax.legend()

fig, ax = plt.subplots()
ax.plot(x_vals,pct_of_time_afp_better_fp[1:])
ax.set_ylabel("Proportion")
ax.set_xlabel("Best responses calculated")
ax.set_title("Proportion of timesteps where AFP is better than FP\n(random 30x30 matrices)")

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
