import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

def build_car_rental_dynamics():
    # parameters
    CREDIT = 10
    MOVING_COST = 2
    LAMBDA_RENTAL_REQUEST_1 = 3
    LAMBDA_RENTAL_REQUEST_2 = 4
    LAMBDA_RETURN_1 = 3
    LAMBDA_RETURN_2 = 2
    MAX_CARS = 20
    MAX_MOVE = 5
    EPSILON = 1e-4

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RENTAL_REQUEST_1) < EPSILON:
            MAX_RENTAL_REQUEST_1 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RENTAL_REQUEST_2) < EPSILON:
            MAX_RENTAL_REQUEST_2 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RETURN_1) < EPSILON:
            MAX_RETURN_1 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RETURN_2) < EPSILON:
            MAX_RETURN_2 = i
            break

    # initialize state space, action space, dynamics, and probability dictionary
    state_space = [(i, j) for i in range(MAX_CARS + 1)
                   for j in range(MAX_CARS + 1)]
    action_space = list(range(-MAX_MOVE, MAX_MOVE + 1))
    dynamics = {}
    prob_dict = {}

    # build probability dictionary
    for rental_request_1 in range(MAX_RENTAL_REQUEST_1 + 1):
        prob_dict[rental_request_1, LAMBDA_RENTAL_REQUEST_1] = poisson.pmf(
            rental_request_1, LAMBDA_RENTAL_REQUEST_1)
    for rental_request_2 in range(MAX_RENTAL_REQUEST_2 + 1):
        prob_dict[rental_request_2, LAMBDA_RENTAL_REQUEST_2] = poisson.pmf(
            rental_request_2, LAMBDA_RENTAL_REQUEST_2)
    for return_1 in range(MAX_RETURN_1 + 1):
        prob_dict[return_1, LAMBDA_RETURN_1] = poisson.pmf(
            return_1, LAMBDA_RETURN_1)
    for return_2 in range(MAX_RETURN_2 + 1):
        prob_dict[return_2, LAMBDA_RETURN_2] = poisson.pmf(
            return_2, LAMBDA_RETURN_2)

    # build dynamics
    for state in state_space:
        for action in action_space:
            dynamics[state, action] = {}
            # invalid action
            if not ((0 <= action <= state[0]) or (-state[1] <= action <= 0)):
                reward = -np.inf
                next_state = state
                dynamics[state, action][next_state, reward] = 1
                continue

            for rental_request_1 in range(MAX_RENTAL_REQUEST_1 + 1):
                for rental_request_2 in range(MAX_RENTAL_REQUEST_2 + 1):
                    for return_1 in range(MAX_RETURN_1 + 1):
                        for return_2 in range(MAX_RETURN_2 + 1):
                            # moving cars
                            next_state = (
                                min(state[0] - action, MAX_CARS), min(state[1] + action, MAX_CARS))
                            reward = -MOVING_COST * abs(action)

                            prob = prob_dict[rental_request_1, LAMBDA_RENTAL_REQUEST_1] * \
                                prob_dict[rental_request_2, LAMBDA_RENTAL_REQUEST_2] * \
                                prob_dict[return_1, LAMBDA_RETURN_1] * \
                                prob_dict[return_2, LAMBDA_RETURN_2]
                            valid_rental_1 = min(next_state[0], rental_request_1)
                            valid_rental_2 = min(next_state[1], rental_request_2)
                            reward = reward + CREDIT * \
                                (valid_rental_1 + valid_rental_2)
                            next_state = (
                                next_state[0] - valid_rental_1, next_state[1] - valid_rental_2)

                            # return cars
                            next_state = (min(next_state[0] + return_1, MAX_CARS), min(
                                next_state[1] + return_2, MAX_CARS))

                            if (next_state, reward) in dynamics[state, action]:
                                dynamics[state, action][next_state, reward] += prob
                            else:
                                dynamics[state, action][next_state, reward] = prob

    init_value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    init_policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    return dynamics, state_space, action_space, init_value, init_policy

def value_iteration(dynamics, state_space, action_space, value, policy, theta=1e-4, gamma=0.9):
    # initialize value
    delta = np.inf
    k = 0
    while delta >= theta:
        k = k + 1
        value_old = value.copy()
        for state in state_space:
            # Update V[s].
            value[state] = max([sum([prob * (reward + gamma * value_old[next_state]) for (
                next_state, reward), prob in dynamics[state, action].items()]) for action in action_space])
            # print('State {}, value = {}'.format(state, value[state]))
        delta = np.max(np.abs(value - value_old))
        print('Iteration {}, delta = {}'.format(k, delta))

    for state in state_space:
        q_max_value = -np.inf
        for action in action_space:
            q_value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if q_value_temp > q_max_value:
                q_max_value = q_value_temp
                policy[state] = action
    return value, policy


if __name__ == '__main__':

    dynamics, state_space, action_space, init_value, init_policy = build_car_rental_dynamics()

    value, policy = value_iteration(
        dynamics, state_space, action_space, init_value, init_policy)

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(policy, cmap='viridis', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('# of cars at second location', fontsize=16)
    ax.set_ylabel('# of cars at first location', fontsize=16)
    ax.set_title('Policy', fontsize=20)
    plt.savefig('policy.png')
    plt.close()

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(value, cmap='viridis', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('# of cars at second location', fontsize=16)
    ax.set_ylabel('# of cars at first location', fontsize=16)
    ax.set_title('Value', fontsize=20)
    plt.savefig('Value.png')
    plt.close()