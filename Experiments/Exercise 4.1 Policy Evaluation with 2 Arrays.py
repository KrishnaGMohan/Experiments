import sys
print(sys.path)

import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

# try out the gw.get_transitions(state, action) function

state = 2
actions = gw.get_available_actions(state)

for action in actions:
    transitions = gw.get_transitions(state=state, action=action)

    # examine each return transition (only 1 per call for this MDP)
    for (trans) in transitions:
        next_state, reward, probability = trans    # unpack tuple
        print("transition("+ str(state) + ", " + action + "):", "next_state=", next_state, ", reward=", reward, ", probability=", probability)


def policy_eval_two_arrays(state_count, gamma, theta, get_policy, get_transitions):
    """
    This function uses the two-array approach to evaluate the specified policy for the specified MDP:
    
    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    
    'gamma' is the MDP discount factor for rewards.
    
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    
    'get_policy' is the stochastic policy function - it takes a state parameter and returns list of tuples, 
        where each tuple is of the form: (action, probability).  It represents the policy being evaluated.
        
    'get_transitions' is the state/reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
        
    """
    V = state_count*[0]
    #
    # INSERT CODE HERE to evaluate the policy using the 2 array approach 
    
    while True:
        delta = 0
        V_prev = list(V)
        V = state_count*[0]
        for state in range(state_count):
            actions = gw.get_available_actions(state)
            policy = dict(get_policy(state))
            for action in actions:
                transitions = gw.get_transitions(state=state, action=action)
                for (trans) in transitions:
                    next_state, reward, probability = trans    # unpack tuple
                    V[state] = V[state] + policy[action] * (probability * (reward + gamma * V_prev[next_state]))
            delta = max(delta,abs(V[state] - V_prev[state]))
        if delta < theta:
            break
    #
    return V

def get_equal_policy(state):
    # build a simple policy where all 4 actions have the same probability, ignoring the specified state
    policy = ( ("up", .25), ("right", .25), ("down", .25), ("left", .25))
    return policy

n_states = gw.get_state_count()

# test our function
values = policy_eval_two_arrays(state_count=n_states, gamma=.9, theta=.001, get_policy=get_equal_policy, \
    get_transitions=gw.get_transitions)

print("Values=", values)


import numpy as np
a = np.append(values, 0)
np.reshape(a, (4,4))

# test our function using the test_db helper
test_dp.policy_eval_two_arrays_test( policy_eval_two_arrays ) 
