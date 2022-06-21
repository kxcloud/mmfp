import numpy as np

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

def get_matching_pennies_abstain(bonus):
    return np.array([
        [1,-1, -bonus],
        [-1,1, -bonus],
        [bonus, bonus, 0]
    ])

def get_cyclic_game(n):
    """
    Return a payoff matrix for a cyclic game. 
        3 -> RPS
    """
    game = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i == (j-1) % n:
                game[i,j] += -1
            if i == (j+1) % n:
                game[i,j] += 1
            
    return game

np.random.seed(2)

game_dict = {
    "Matching Pennies" : np.array([[1,-1],[-1,1]]),
    "Matching Pennies Abstain" : get_matching_pennies_abstain(bonus=0.05),
    "RPS" : np.array([[0,-1,1],[1,0,-1],[-1,1,0]]),
    "Biased RPS" : np.array([[0,-1,2],[1,0,-1],[-1,1,0]]),
    "weakRPS" : np.array([[0,-1,1e-1],[1,0,-1],[-1e-1,1,0]]),
    "RPS + safe R" : np.array([[0,-1,1,0],[1,0,-1,0.1],[-1,1,0,-0.9],[0,-0.1,0.9,0]]),
    "RPS Abstain": get_rps_abstain(bonus=0.05),
    "Random game 1" : np.array([[ 1.62, -0.61, -0.53],
                                [-1.07,  0.87, -2.3 ],
                                [ 1.74, -0.76,  0.32]]),
    "Random game 2" : np.array([[-0.42, -0.06, -2.14],
                                [ 1.64, -1.79, -0.84],
                                [ 0.5,  -1.25, -1.06]]),
    "RPS with mixed moves" : get_rps_with_mixed_moves(bonus=0.1),
    "Albert's RPS + safe R": np.array(
        [
            [ 0, -1,  1, 0.0],
            [ 1,  0, -1, 0.88],
            [-1,  1,  0, -0.9],
            [ 0.0, -0.88, 0.9, 0.0],
        ]),
    "Cyclic game" : get_cyclic_game(6),
  }