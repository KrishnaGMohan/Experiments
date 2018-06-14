
import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

state_count = gw.get_state_count()
pi = state_count*[0]
for s in range(state_count):
    avail_actions = gw.get_available_actions(s)
    pi[s] = avail_actions[0]

avail_actions = gw.get_available_actions(2)
print(avail_actions)
transitions = gw.get_transitions(state=2, action=pi[2])
print(transitions)
next_state, reward, probability = transitions[0]
print(next_state, reward, probability)

for s in range(state_count):
    avail_actions = gw.get_available_actions(s)
    print(avail_actions)
    pi[s] = avail_actions[0]
    print(len(avail_actions))

print(pi)
