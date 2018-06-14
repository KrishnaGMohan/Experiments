import numpy as np
shape = (4, 12)
nS = np.prod(shape)
nA = 4

_cliff = np.zeros(shape, dtype=np.bool)
_cliff[3, 1:-1] = True
print(_cliff)

P = {}

def _limit_coordinates(coord):
    coord[0] = min(coord[0], shape[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], shape[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

def _calculate_transition_prob(current, delta):
    new_position = np.array(current) + np.array(delta)
    new_position = _limit_coordinates(new_position).astype(int)
    new_state = np.ravel_multi_index(tuple(new_position), shape)
    reward = -100.0 if _cliff[tuple(new_position)] else -1.0
    is_done = _cliff[tuple(new_position)] or (tuple(new_position) == (3,11))
    return [(1.0, new_state, reward, is_done)]

for s in range(nS):
    position = np.unravel_index(s, shape)
    print(position)
    P[s] = { a : [] for a in range(nA) }
    #UP = 0
    #RIGHT = 1
    #DOWN = 2
    #LEFT = 3
    P[s][0] = _calculate_transition_prob(position, [-1, 0])
    P[s][1] = _calculate_transition_prob(position, [0, 1])
    P[s][2] = _calculate_transition_prob(position, [1, 0])
    P[s][3] = _calculate_transition_prob(position, [0, -1])
    print(P[s])


