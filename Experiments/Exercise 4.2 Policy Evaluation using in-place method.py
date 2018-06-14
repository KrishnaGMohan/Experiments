import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld


def policy_eval_in_place(state_count, gamma, theta, get_policy, get_transitions):
    V = state_count*[0]
        
    while True:
        delta = 0
        for state in range(state_count):
            actions = gw.get_available_actions(state)
            policy = dict(get_policy(state))
            v = V[state]
            v_new = 0
            for action in actions:
                transitions = gw.get_transitions(state=state, action=action)
                for (trans) in transitions:
                    next_state, reward, probability = trans    # unpack tuple
                    v_new = v_new + policy[action] * (probability * (reward + gamma * V[next_state]))
            V[state] = v_new
            delta = max(delta, abs(v - V[state]))
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
values = policy_eval_in_place(state_count=n_states, gamma=.9, theta=.001, get_policy=get_equal_policy, \
    get_transitions=gw.get_transitions)

print("Values=", values)

# test our function using the test_db helper
test_dp.policy_eval_in_place_test(policy_eval_in_place)