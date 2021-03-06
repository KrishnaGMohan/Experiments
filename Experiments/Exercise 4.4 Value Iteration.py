import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

def value_iteration(state_count, gamma, theta, get_available_actions, get_transitions):
    """
    This function computes the optimal value function and policy for the specified MDP, using the Value Iteration algorithm.
    
    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    
    'gamma' is the MDP discount factor for rewards.
    
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    
    'get_available_actions' returns a list of the MDP available actions for the specified state parameter.
    
    'get_transitions' is the MDP state / reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
    """
    # init all state value estimates to 0
    V = state_count*[0]                
    pi = state_count*[0]
    
    # init with a policy with first avail action for each state
    for s in range(state_count):
        avail_actions = get_available_actions(s)
        pi[s] = avail_actions[0]
    # print("Initial policy", pi)

    while True:
        delta = 0
        V_prev = list(V)
        V = state_count*[-11]

        for state in range(state_count):    
            V_best = V_prev[state]
            actions = gw.get_available_actions(state)
            for action in actions:
                #print("State:", state, "Action:", action)
                transitions = gw.get_transitions(state=state, action=action)
                V_sa = 0
                for (trans) in transitions:
                    next_state, reward, probability = trans    # unpack tuple
                    V_sa = V_sa + probability * (reward + gamma * V_prev[next_state])

                if V_sa > V_best:
                    pi[state] = action
                    V_best = V_sa
                    V[state] = V_sa
            delta = max(delta,abs(V[state] - V_prev[state]))

        if delta < theta:
            break

    V = V_prev
    # insert code here to iterate using policy evaluation and policy improvement (see Policy Iteration algorithm)
    return (V, pi)        # return both the final value function and the final policy


n_states = gw.get_state_count()

# test our function
values, policy = value_iteration(state_count=n_states, gamma=.9, theta=.001, get_available_actions=gw.get_available_actions, \
    get_transitions=gw.get_transitions)

print("Values=", values)
print("Policy=", policy)

# test our function using the test_db helper
test_dp.value_iteration_test( value_iteration ) 


