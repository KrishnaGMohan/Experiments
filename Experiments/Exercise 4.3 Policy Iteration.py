# D
import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

def policy_iteration(state_count, gamma, theta, get_available_actions, get_transitions):
    """
    This function computes the optimal value function and policy for the specified MDP, using the Policy Iteration algorithm.
    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    
    'gamma' is the MDP discount factor for rewards.
    
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    
    'get_available_actions' returns a list of the MDP available actions for the specified state parameter.
    
    'get_transitions' is the MDP state / reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
    """
    # 1.Initialization
    # init all state value estimates to 0
    V = state_count*[0]                
    pi = state_count*[0]
    
    # init with a policy with first avail action for each state
    for s in range(state_count):
        avail_actions = get_available_actions(s)
        pi[s] = avail_actions[0]
    # print("Initial policy", pi)
    
    iterations = 0
    policy_stable = False
    while not policy_stable:
        iterations += 1
                
        # 3. Policy Evaluation
        while True:
            delta = 0
            
            V_prev = list(V)
            V = state_count*[0]
            for state in range(state_count):
                action = pi[state]
                transitions = gw.get_transitions(state=state, action=action)
                for (trans) in transitions:
                    next_state, reward, probability = trans    # unpack tuple
                    V[state] = V[state] + probability * (reward + gamma * V_prev[next_state])
            delta = max(delta,abs(V[state] - V_prev[state]))
            
            if delta < theta:
                break

        # 3. Policy Improvement
        policy_stable = True
        for state in range(state_count):    
            old_action = pi[state]
            
            V_best = V[state]
            actions = gw.get_available_actions(state)
            for action in actions:
                transitions = gw.get_transitions(state=state, action=action)
                V_sa = 0
                for (trans) in transitions:
                    next_state, reward, probability = trans    # unpack tuple
                    V_sa = V_sa + probability * (reward + gamma * V[next_state])
                if V_sa > V_best:
                    pi[state] = action
                    V_best = V_sa

            if old_action != pi[state]:
                policy_stable = False

        #print("Iteration:", iterations)
        #print("Values=", V)
        #print("Policy=", pi)
        
    return (V, pi)        # return both the final value function and the final policy
n_states = gw.get_state_count()

# test our function
values, policy = policy_iteration(state_count=n_states, gamma=.9, theta=.001, get_available_actions=gw.get_available_actions, \
    get_transitions=gw.get_transitions)

print("Values=", values)
print("Policy=", policy)

import numpy as np
a = np.append(values, 0)
np.reshape(a, (4,4))


a = np.append(policy, policy[0])
np.reshape(a, (4,4))


# test our function using the test_db helper
test_dp.policy_iteration_test( policy_iteration ) 