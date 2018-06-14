
import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

state_count = gw.get_state_count()
pi = state_count*[0]

for s in range(state_count):
    avail_actions = gw.get_available_actions(s)
    pi[s] = avail_actions[0]
    print(avail_actions)

print(pi)
