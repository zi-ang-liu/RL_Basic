'''solving the cliff walking problem using linear programming'''
from gurobipy import GRB, Model, quicksum
import gymnasium as gym
import numpy as np

def lp_solver(r, p, gamma):

    n_state = r.shape[0]

    # create a model instance
    model = Model()

    # create variables
    for s in range(n_state):
        model.addVar(name=f'v_{s}', lb=-GRB.INFINITY)
    
    # update the model
    model.update()

    # create constraints
    for state in reachable_state_set:
        for action in action_set:
            model.addConstr(model.getVarByName(f'v_{state}') >= quicksum(
                gamma * p[state, action, next_state] * model.getVarByName(f'v_{next_state}') for next_state in reachable_state_set ) + r[state, action])

    # set objective
    model.setObjective(quicksum(model.getVarByName(
        f'v_{state}') for state in reachable_state_set ), GRB.MINIMIZE)

    # optimize
    model.optimize()

    return model

if __name__ == '__main__':

    # create an environment
    env = gym.make('CliffWalking-v0')
    n_state = env.unwrapped.nS
    n_action = env.unwrapped.nA
    state_set = set(range(n_state))
    action_set = set(range(n_action))
    # The player cannot be at the cliff, nor at the goal 
    terminal_state_set = [47] 
    unreachable_state_set = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    # the reachable state set is the set of all states except the cliff and the goal.
    # only the states in the reachable state set are considered in the optimization problem
    reachable_state_set = set(set(state_set) - set(terminal_state_set) - set(unreachable_state_set))

    # set parameters
    gamma = 1

    # initialize reward and transition probability
    r = np.zeros((n_state, n_action))
    p = np.zeros((n_state, n_action, n_state))

    for state in reachable_state_set:
        for action in action_set:
            for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
                r[state, action] += prob * reward
                p[state, action, next_state] += prob

    # solve the mdp problem using linear programming
    model = lp_solver(r, p, gamma)

    # value
    value_function = {}
    for state in reachable_state_set:
        value_function[state] = model.getVarByName(f'v_{state}').x

    # action value
    action_value_function = {}
    for state in reachable_state_set:
        action_value_function[state] = {}
        for action in action_set:
            action_value_function[state][action] = r[state, action] + gamma * sum(
                p[state, action, next_state] * value_function[next_state] for next_state in reachable_state_set)
            
    # optimal policy
    optimal_policy = {}
    for state in reachable_state_set:
        optimal_policy[state] = max(action_value_function[state], key=action_value_function[state].get)
            
    # print value function 4*12, 1 digital after decimal point

    print('value function = ')
    for i in range(4):
        for j in range(12):
            if i * 12 + j in value_function:
                print('{:.1f}'.format(value_function[i * 12 + j]), end='\t')
            else:
                print('x', end='\t')
        print()

    print('optimal policy = ')
    for i in range(4):
        for j in range(12):
            if i * 12 + j in optimal_policy:
                print(optimal_policy[i * 12 + j], end='\t')
            else:
                print('x', end='\t')
        print()

    model.write("lo1.lp")
