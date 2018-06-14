import numpy as np
import sys

shape = (7, 10)
nA = 4
winds = np.zeros(shape)
winds[:,[3,4,5,8]] = 1
winds[:,[6,7]] = 2

def _limit_coordinates(coord):
    coord[0] = min(coord[0], shape[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], shape[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

def _calculate_transition_prob(current, delta, winds):
    new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
    new_position = _limit_coordinates(new_position).astype(int)
    new_state = np.ravel_multi_index(tuple(new_position), shape)
    is_done = tuple(new_position) == (3, 7)
    return [(1.0, new_state, -1.0, is_done)]

position = (3,6)

P = _calculate_transition_prob(position, [-1, 0], winds)
print(P)
s = P[0][1]
print(np.unravel_index(s, shape))

P = { a : [] for a in range(nA) }
#UP = 0
#RIGHT = 1
#DOWN = 2
#LEFT = 3
P[0] = _calculate_transition_prob(position, [-1, 0], winds)
P[1] = _calculate_transition_prob(position, [0, 1], winds)
P[2] = _calculate_transition_prob(position, [1, 0], winds)
P[3] = _calculate_transition_prob(position, [0, -1], winds)

print(P)