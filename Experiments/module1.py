previous_action = None        
for i in range(10):
    if previous_action is None:
        current_action = 1
    else:
        current_action = previous_action % 3 + 1
    print(previous_action, current_action)
    previous_action = current_action

import numpy as np
num_actions = 5
initial_value = 5
round = 6
total_rewards = np.array([2, 3, 1, 3, 4])
total_counts = np.full(num_actions, initial_value, dtype = np.longdouble)
print(total_rewards)
print(total_counts)
current_averages = np.divide(total_rewards,total_counts, where = total_counts > 0) 
rounds = np.full(num_actions, round, dtype = np.longdouble)
x = np.sqrt(2 * np.log(rounds)/total_counts)
print(current_averages + x)


x = np.sqrt(np.divide(2 * rounds, np.log(, total_counts))




import numpy as np
x = np.array([2,3,1,0])
y = np.array([2,3,1,0])
print(x + y)
