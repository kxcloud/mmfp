import matplotlib.pyplot as plt
import numpy as np

import lp_solver

import matplotlib.style as style
style.use('tableau-colorblind10')

def one_hot(index, length):
    array = np.zeros(length)
    array[index] = 1
    return array

def one_hot_argmax(array):
    one_hot = np.zeros_like(array)
    one_hot[np.argmax(array)] = 1
    return one_hot

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

class IterativePlayer:
    """
    Store data from a run of a Fictitious-Play-type algorithm. 
    """
    def __init__(self, game, t_max, initial_action_p1=0, initial_action_p2=1):
        self.game = game
        
        self.p1_response = np.zeros((t_max, game.shape[0]))
        self.p2_response = np.zeros((t_max, game.shape[1]))
        self.p1_response[0,initial_action_p1] = 1
        self.p2_response[0,initial_action_p2] = 1

        self.p1_empirical = self.p1_response.copy()
        self.p2_empirical = self.p2_response.copy()
        
        self.worst_case_payoff = np.zeros((t_max, 2))
        self.worst_case_payoff[0,:] = None
        
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
        assert abs(np.sum(p1_strategy) - 1) < 1e-4, f"Probs must sum to 1. {p1_strategy}"
        assert abs(np.sum(p2_strategy) - 1) < 1e-4, f"Probs must sum to 1. {p2_strategy}"
        
        self.p1_response[self.t,:] = p1_strategy
        self.p2_response[self.t,:] = p2_strategy
        
        self.p1_empirical[self.t,:] = np.mean(self.p1_response[:self.t+1,], axis=0)
        self.p2_empirical[self.t,:] = np.mean(self.p2_response[:self.t+1,], axis=0)
        
        self.worst_case_payoff[self.t,0] = np.min(self.p1_empirical[self.t] @ self.game)
        self.worst_case_payoff[self.t,1] = -np.max(self.game @ self.p2_empirical[self.t])
        
        self.t += 1
    
    def plot(self, title="", players_to_plot=[0,1], figsize=(12,8)):   
        
        fig, axes = plt.subplots(
            nrows=2, ncols=len(players_to_plot), sharex=True, 
            sharey="row", figsize=figsize, squeeze=False
        )
        
        if 0 in players_to_plot:
            plot_single_player(self.p1_response, self.p1_empirical, axes[0,0], "Player 1")
        if 1 in players_to_plot:
            plot_single_player(self.p2_response, self.p2_empirical, axes[0,1], "Player 2")
        
        for player_idx in players_to_plot:
            ax = axes[1,player_idx]
            ax.plot(self.worst_case_payoff[:,player_idx], lw=2, c="grey")
        
        for player_idx, nash_probs in enumerate([self.p1_probs_nash, self.p2_probs_nash]):
            for action_idx, prob in enumerate(nash_probs):
                color = f"C{action_idx}"
                axes[0, player_idx].axhline(prob, ls="--", c=color, lw=1)
                    
        axes[1,0].axhline(self.value, label="value", ls="--", c="black", lw=0.8)
        axes[1,1].axhline(-self.value, label="value", ls="--", c="black", lw=0.8)
        axes[0,0].set_ylabel("Strategy probabilities")
        axes[1,0].set_ylabel("Worst-case payoff")
        axes[-1,0].set_xlabel("Timestep")
        
        axes[1,1].legend()
        plt.suptitle(title)       
        return axes

if __name__ == "__main__":
    np.random.seed(48)
    
    games = {
        "Matching Pennies" : np.array([[1,-1],[-1,1]]),
        "RPS" : np.array([[0,-1,1],[1,0,-1],[-1,1,0]]),
        "Biased RPS" : np.array([[0,-1,2],[1,0,-1],[-1,1,0]]),
        "RPS + safe R" : np.array([[0,-1,1,0],[1,0,-1,0.1],[-1,1,0,-0.9],[0,-0.1,0.9,0]]),
        "Random game" : np.random.normal(size=(5,5))
    }
    game_name = "Biased RPS"
    # game_name = "Matching Pennies"
    # game_name = "RPS + safe R"
    # game_name = "Random game"
    # game_name = "RPS"
    
    game = games[game_name]
    p1_first_action = 0
    p2_first_action = 1
        
    # Fictitious Play
    t_max = 150
    play = IterativePlayer(game, t_max, p1_first_action, p2_first_action)
        
    for t in range(1, t_max):
        play.add_strategies(
            one_hot_argmax(play.game @ play.p2_empirical[t-1]),
            one_hot_argmax(-play.game.T @ play.p1_empirical[t-1])
        )
        
    play.plot(title=f"Fictitious Play: {game_name}")
    
    # Maximin Fictitious Play
    
    p1_restricted_values = []
    p2_restricted_values = []
    
    play = IterativePlayer(game, t_max, p1_first_action, p2_first_action)
    for t in range(1, t_max):
        # p1_last = play.p1_response[t-1]
        # p2_last = play.p2_response[t-1]
        p1_last = one_hot(0, game.shape[0])
        p2_last = one_hot(0, game.shape[1])
                
        p1_restricted_game = play.game @ np.array([play.p2_empirical[t-1], p2_last]).T
        p2_restricted_game = -play.game.T @ np.array([play.p1_empirical[t-1], p1_last]).T
        
        p1_probs, lp_res_1 = lp_solver.solve_game_half(-p1_restricted_game.T)
        p2_probs, lp_res_2 = lp_solver.solve_game_half(-p2_restricted_game.T)
        
        p2_restricted_values.append(lp_res_1["res"]["fun"])
        p1_restricted_values.append(lp_res_2["res"]["fun"])
        
        # if np.abs(p1_probs[0] - 1) < 1e-5 and t > 50:
        #     breakpoint()
        
        play.add_strategies(
            p1_probs,
            p2_probs
        )
        
    axes = play.plot(title=f"Maximin Fictitious Play: {game_name}")
    axes[1,0].plot(p1_restricted_values, label="restricted game value", ls=":")
    axes[1,1].plot(p2_restricted_values, label="restricted game value", ls=":")
    axes[1,1].legend()
    
    # Maximin Fictitious Play v2
    play = IterativePlayer(game, t_max, p1_first_action, p2_first_action)
    play.add_strategies(one_hot(p1_first_action, game.shape[0]), one_hot(p2_first_action, game.shape[1]))
    play.add_strategies(one_hot(p1_first_action, game.shape[0]), one_hot(p2_first_action, game.shape[1]))
    
    for t in range(3, t_max):   
        p1_restricted_game = play.game @ play.p2_empirical[[t-2,t-1]].T
        p2_restricted_game = -play.game.T @ play.p1_empirical[[t-2,t-1]].T
        
        p1_probs, _ = lp_solver.solve_game_half(-p1_restricted_game.T)
        p2_probs, _ = lp_solver.solve_game_half(-p2_restricted_game.T)
        
        play.add_strategies(
            p1_probs,
            p2_probs
        )
        
    play.plot(title=f"Maximin Fictitious Play v2: {game_name}")
