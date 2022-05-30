import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import ternary

import lp_solver

import matplotlib.style as style
style.use('tableau-colorblind10')

def one_hot(index, length):
    array = np.zeros(length)
    array[index] = 1
    return array

def get_rps_with_mixed_moves(bonus=0):
    rps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    
    moves = [[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
    bonuses = [0,0,0, bonus, bonus, bonus]
    
    rps_with_mixed_moves = np.zeros((6,6))
    for i, (move_1, bonus_1) in enumerate(zip(moves,bonuses)):
        for j, (move_2, bonus_2) in enumerate(zip(moves,bonuses)):
            rps_with_mixed_moves[i,j] = move_1 @ rps @ move_2 + bonus_1 - bonus_2
    return rps_with_mixed_moves
    
def get_rps_abstain(bonus=0):
    rps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    moves = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    bonuses = [0,0,0, bonus]
    
    rps_abstain = np.zeros((4,4))
    for i, (move_1, bonus_1) in enumerate(zip(moves,bonuses)):
        for j, (move_2, bonus_2) in enumerate(zip(moves,bonuses)):
            rps_abstain[i,j] = move_1 @ rps @ move_2 + bonus_1 - bonus_2
    return rps_abstain

def normalize_probabilities(array):
    array = np.clip(array, 0, np.inf)
    if np.sum(array) < 1e-6:
        array = array + 1e-5
    array = array / array.sum()
    return array

def one_hot_argmax(array, noise=None):
    one_hot = np.zeros_like(array)
    one_hot[np.argmax(array)] = 1
    
    if noise is None:
        return one_hot
    
    noise_vector = np.random.normal(size=len(array), scale=noise)
    noisy_one_hot = normalize_probabilities(one_hot + noise_vector)
    return noisy_one_hot
    
def plot_single_player(history, empirical_history, ax=None, title=""):
    lw = 1.5 # line width
    
    ax = ax or plt.subplots()[1] 
    t_max, num_strategies = history.shape

    for strategy_idx in range(num_strategies):
        color = f"C{strategy_idx}"
        
        action_inds = history[:, strategy_idx]
        action_inds[action_inds == 0] = np.nan
        
        ax.scatter(range(t_max), action_inds, lw=0.1, c=color, s=15)    
        ax.plot(empirical_history[:,strategy_idx], lw=lw, ls="-")
    
    ax.set_title(title)

def plot_on_triangle(iterative_player, title=None):
    # https://github.com/marcharper/python-ternary
    
    fig, axes = plt.subplots(ncols=2, figsize=(12,6))
    
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
    
    tax_p1 = ternary.TernaryAxesSubplot(ax=axes[0])
    tax_p2 = ternary.TernaryAxesSubplot(ax=axes[1])

    tax_p1.plot(np.nan_to_num(iterative_player.p1_empirical))
    tax_p2.plot(np.nan_to_num(iterative_player.p2_empirical))

    tax_p1.scatter([iterative_player.p1_probs_nash], marker="*", c="black", s=60, zorder=10)
    tax_p2.scatter([iterative_player.p2_probs_nash], marker="*", c="black", s=60, zorder=10)
    
    tax_p1.set_title("Player 1 strategies")
    tax_p2.set_title("Player 2 strategies")
    
    return axes

class IterativePlayer:
    """
    Store data from a run of a Fictitious-Play-type algorithm. 
    """
    def __init__(self, game, t_max, initial_strategy_p1, initial_strategy_p2):
        # Payoff matrix for two-player zero-sum game.
        self.game = game
        
        assert (len(initial_strategy_p1), len(initial_strategy_p2)) == self.game.shape, (
            "Strategy sizes don't match game shape"    
        )
        for strategy in [initial_strategy_p1, initial_strategy_p2]:
            assert np.isclose(np.sum(strategy), 1, atol=1e-4), f"Probs must sum to 1. {strategy}"
        
        self.p1_response = np.zeros((t_max, game.shape[0]))
        self.p2_response = np.zeros((t_max, game.shape[1]))
        self.p1_response[0,:] = initial_strategy_p1
        self.p2_response[0,:] = initial_strategy_p2

        self.p1_empirical = self.p1_response.copy()
        self.p2_empirical = self.p2_response.copy()
        
        self.worst_case_payoff = np.zeros((t_max, 2))
        self.worst_case_payoff[0,:] = None
        
        # Minimax Theorem (von Neumann, 1928): the set of Nash equilibria
        # is { (x*, y*) : x* is maximin for P1, y* is maximin for P2 }.
        # Informally, 2P0S => Nash = Maximin = Minimax.
        self.p2_probs_nash, info = lp_solver.solve_game_half(game)
        self.p1_probs_nash, _   = lp_solver.solve_game_half(-game.T)
        self.value = info["res"]["fun"]
        
        self.t = 1
    
    def add_strategies(self, p1_strategy, p2_strategy):
        """ 
        The strategies here are the ones added by a FP-like algorithm.
        
        This function simply updates the data
        """
        assert (len(p1_strategy), len(p2_strategy)) == self.game.shape, (
            "Strategy sizes don't match game shape"    
        )
        assert np.isclose(np.sum(p1_strategy), 1, atol=1e-4), f"Probs must sum to 1. {p1_strategy}"
        assert np.isclose(np.sum(p2_strategy), 1, atol=1e-4), f"Probs must sum to 1. {p2_strategy}"
        
        self.p1_response[self.t,:] = p1_strategy
        self.p2_response[self.t,:] = p2_strategy
        
        self.p1_empirical[self.t,:] = np.mean(self.p1_response[:self.t+1,], axis=0)
        self.p2_empirical[self.t,:] = np.mean(self.p2_response[:self.t+1,], axis=0)
        
        self.worst_case_payoff[self.t] = self.worst_case_payoffs(
            self.p1_empirical[self.t], self.p2_empirical[self.t]
        )
        
        self.t += 1
    
    def worst_case_payoffs(self, p1_strategy, p2_strategy):
        worst_case_p1 = np.min(p1_strategy @ self.game)
        worst_case_p2 = -np.max(self.game @ p2_strategy)
        return worst_case_p1, worst_case_p2
    
    def plot(self, title="", players_to_plot=[0,1], figsize=(12,8)):   
        
        fig, axes = plt.subplots(
            nrows=2, ncols=len(players_to_plot), sharex=True, 
            sharey="row", figsize=figsize, squeeze=False
        )
        
        axes[1,0].axhline(self.value, label="value of game", ls="--", c="black", lw=0.8)
        axes[1,1].axhline(-self.value, label="value of game", ls="--", c="black", lw=0.8)
        
        if 0 in players_to_plot:
            plot_single_player(self.p1_response, self.p1_empirical, axes[0,0], "Player 1")
        if 1 in players_to_plot:
            plot_single_player(self.p2_response, self.p2_empirical, axes[0,1], "Player 2")
        
        for player_idx in players_to_plot:
            ax = axes[1,player_idx]
            ax.plot(self.worst_case_payoff[:,player_idx], lw=2, c="grey", label="average strategy")
        
        for player_idx, nash_probs in enumerate([self.p1_probs_nash, self.p2_probs_nash]):
            for action_idx, prob in enumerate(nash_probs):
                color = f"C{action_idx}"
                axes[0, player_idx].axhline(prob, ls="--", c=color, lw=1)
                    
        axes[0,0].set_ylabel("Strategy probabilities")
        axes[1,0].set_ylabel("Worst-case payoff")
        axes[-1,0].set_xlabel("Timestep")
        
        axes[1,1].legend()
        plt.suptitle(title)       
        return axes

if __name__ == "__main__":
    seed = np.random.choice(10000)
    print(f"Seed: {seed}")
    np.random.seed(seed)
    
    games = {
        "Matching Pennies" : np.array([[1,-1],[-1,1]]),
        "RPS" : np.array([[0,-1,1],[1,0,-1],[-1,1,0]]),
        "Biased RPS" : np.array([[0,-1,2],[1,0,-1],[-1,1,0]]),
        "RPS + safe R" : np.array([[0,-1,1,0],[1,0,-1,0.1],[-1,1,0,-0.9],[0,-0.1,0.9,0]]),
        "RPS Abstain": get_rps_abstain(bonus=0.05),
        "Random game" : np.random.normal(size=(9,9)),
        "RPS with mixed moves" : get_rps_with_mixed_moves(bonus=0.1),
        "Albert's RPS + safe R": np.array(
            [
                [ 0, -1,  1, 0.0],
                [ 1,  0, -1, 0.88],
                [-1,  1,  0, -0.9],
                [ 0.0, -0.88, 0.9, 0.0],
            ])
      }
    # game_name = "Biased RPS"
    # game_name = "Matching Pennies"
    # game_name = "RPS + safe R"
    # game_name = "Random game"
    # game_name = "RPS with mixed moves"
    # game_name = "RPS Abstain"
    game_name = "RPS"
    # game_name = "Albert's RPS + safe R"
    
    game = games[game_name]
    
    initial_strategy_p1 = one_hot(0, game.shape[0]) #np.ones(game.shape[0]) / game.shape[0]
    initial_strategy_p2 = one_hot(0, game.shape[1]) #np.ones(game.shape[1]) / game.shape[1]
    t_max = 100
        
    # FICTITIOUS PLAY
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
        
    for t in range(1, t_max):
        play.add_strategies(
            one_hot_argmax(play.game @ play.p2_empirical[t-1]),
            one_hot_argmax(-play.game.T @ play.p1_empirical[t-1])
        )
        
    play.plot(title=f"Fictitious Play: {game_name}")
    # plot_on_triangle(play)
    
    # FICTITIOUS PLAY (noisy)
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
        
    # for t in range(1, t_max):
    #     noise = 1/(np.log(t) + 1)
    #     play.add_strategies(
    #         one_hot_argmax(play.game @ play.p2_empirical[t-1], noise=noise),
    #         one_hot_argmax(-play.game.T @ play.p1_empirical[t-1], noise=noise)
    #     )
        
    # play.plot(title=f"Fictitious Play (noisy): {game_name}")
    
    # ANTICIPATORY FICTITIOUS PLAY (V2)
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    
    for t in range(1, t_max):
        noise = 0 #1/(np.log(t) + 1)
        p1_payoffs = play.game @ play.p2_empirical[t-1]* (t-1)
        p2_payoffs = -play.game.T @ play.p1_empirical[t-1] * (t-1)
        
        p1_br = one_hot_argmax(p1_payoffs, noise=noise)
        p2_br = one_hot_argmax(p2_payoffs, noise=noise)
        
        p1_br = one_hot(0, game.shape[0])
        p1_br = one_hot(0, game.shape[1])
        
        p1_ar = one_hot_argmax(p1_payoffs + play.game @ p2_br, noise=noise)
        p2_ar = one_hot_argmax(p2_payoffs + -play.game.T @ p1_br, noise=noise)
        
        play.add_strategies(
            p1_ar, p2_ar
        )
        
    play.plot(title=f"Anticipatory Fictitious Play (v2): {game_name}")
    plot_on_triangle(play)
    
    # # MAXIMIN FICTITIOUS PLAY
    # p1_restricted_values = []
    # p2_restricted_values = []
    
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    # for t in range(1, t_max):
    #     p1_last = play.p1_response[t-1]
    #     p2_last = play.p2_response[t-1]
    #     # p1_last = one_hot(np.random.choice(game.shape[0]), game.shape[0])
    #     # p2_last = one_hot(np.random.choice(game.shape[1]), game.shape[1])
    #     # p1_last = one_hot(0, game.shape[0])
    #     # p2_last = one_hot(0, game.shape[1])
                
    #     p1_restricted_game = play.game @ np.array([play.p2_empirical[t-1], p2_last]).T
    #     p2_restricted_game = -play.game.T @ np.array([play.p1_empirical[t-1], p1_last]).T
        
    #     p1_probs, lp_res_1 = lp_solver.solve_game_half(-p1_restricted_game.T)
    #     p2_probs, lp_res_2 = lp_solver.solve_game_half(-p2_restricted_game.T)
        
    #     p2_restricted_values.append(lp_res_1["res"]["fun"])
    #     p1_restricted_values.append(lp_res_2["res"]["fun"])
                
    #     play.add_strategies(
    #         p1_probs,
    #         p2_probs
    #     )
        
    # axes = play.plot(title="Maximin Fictitious Play: " f"{game_name}")
    # axes[1,0].plot(p1_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].plot(p2_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].legend()
    
    # MAXIMIN FICTITIOUS PLAY (counter example)
    # p1_restricted_values = []
    # p2_restricted_values = []
    
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    # for t in range(1, t_max):
    #     # p1_last = play.p1_response[t-1]
    #     # p2_last = play.p2_response[t-1]
    #     # p1_last = one_hot(np.random.choice(game.shape[0]), game.shape[0])
    #     # p2_last = one_hot(np.random.choice(game.shape[1]), game.shape[1])
    #     p1_last = play.p1_empirical[int((t-1)/2)]
    #     p2_last = play.p2_empirical[int((t-1)/2)]
                
    #     p2_strategy_set = np.array([play.p2_empirical[t-1], p2_last]).T
    #     p1_strategy_set = np.array([play.p1_empirical[t-1], p1_last]).T
        
    #     p1_restricted_game = play.game @ p2_strategy_set
    #     p2_restricted_game = -play.game.T @ p1_strategy_set
        
    #     p1_probs, lp_res_1 = lp_solver.solve_game_half(-p1_restricted_game.T)
    #     p2_probs, lp_res_2 = lp_solver.solve_game_half(-p2_restricted_game.T)
        
    #     p2_restricted_values.append(lp_res_1["res"]["fun"])
    #     p1_restricted_values.append(lp_res_2["res"]["fun"])
                
    #     play.add_strategies(
    #         p1_probs,
    #         p2_probs
    #     )
        
    # axes = play.plot(title="Maximin Fictitious Play (variant): " f"{game_name}")
    # axes[1,0].plot(p1_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].plot(p2_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].legend()
    
    # # ANYTIME PSRO 
    # p1_restricted_values = []
    # p2_restricted_values = []
    
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)

    # for t in range(1, t_max):   
    #     p1_restricted_game = play.game @ play.p2_response[:t].T
    #     p2_restricted_game = -play.game.T @ play.p1_response[:t].T
        
    #     p1_probs, lp_res_1 = lp_solver.solve_game_half(-p1_restricted_game.T)
    #     p2_probs, lp_res_2 = lp_solver.solve_game_half(-p2_restricted_game.T)
        
    #     p2_restricted_values.append(lp_res_1["res"]["fun"])
    #     p1_restricted_values.append(lp_res_2["res"]["fun"])
        
    #     play.add_strategies(
    #         p1_probs,
    #         p2_probs
    #     )
        
    # axes = play.plot(title=f"Anytime PSRO: {game_name}")
    
    # axes[1,0].plot(p1_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].plot(p2_restricted_values, label="restricted game value", ls=":")
    # axes[1,1].legend()
    
    # Simple algorithm
    # p1_response_values = [None]
    # p2_response_values = [None]
    
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    
    # for t in range(1, t_max):
    #     gamma = np.clip(np.log(t+5)/t,0,1)
    #     p1_last = play.p1_response[t-1]
    #     p2_last = play.p2_response[t-1]
        
    #     p1_next = (1-gamma)*p1_last + gamma * one_hot_argmax(t*play.game @ p2_last)
    #     p2_next = (1-gamma)*p2_last + gamma * one_hot_argmax(-t*p1_last @ play.game)
        
    #     p1_worst, p2_worst = play.worst_case_payoffs(p1_next, p2_next)
    #     p1_response_values.append(p1_worst)
    #     p2_response_values.append(p2_worst)
        
    #     play.add_strategies(
    #         p1_next, p2_next
    #     )
        
    # axes = play.plot(title=f"Naive updating: {game_name}")
    
    # axes[1,0].plot(p1_response_values, ls=":")
    # axes[1,1].plot(p2_response_values, label="response", ls=":")
    # axes[1,1].legend()
    
    # plot_on_triangle(play)
    
    # # Ryan's alg
    # p1_response_values = [None]
    # p2_response_values = [None]
    
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    
    # for t in range(1, t_max):
    #     gamma = 1/(t+1)
    #     p1_last = play.p1_response[t-1]
    #     p2_last = play.p2_response[t-1]
        
    #     update_x = p1_last * np.exp(t * play.game @ p2_last) 
    #     update_y = p2_last * np.exp(-t * play.game.T @ p1_last)
        
    #     p1_next = (1-gamma)*p1_last + gamma * update_x / update_x.sum()
    #     p2_next = (1-gamma)*p2_last + gamma * update_y / update_y.sum()
        
    #     p1_worst, p2_worst = play.worst_case_payoffs(p1_next, p2_next)
    #     p1_response_values.append(p1_worst)
    #     p2_response_values.append(p2_worst)
        
    #     play.add_strategies(
    #         p1_next, p2_next
    #     )
        
    # axes = play.plot(title=f"Ryan's alg': {game_name}")
    
    # axes[1,0].plot(p1_response_values, ls=":")
    # axes[1,1].plot(p2_response_values, label="strategy", ls=":")
    # axes[1,1].legend()
    
    # plot_on_triangle(play)