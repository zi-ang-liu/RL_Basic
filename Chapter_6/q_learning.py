import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def q_learning(alpha, epsilon, gamma, num_episodes, env):
    # initialize Q(s,a)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # initialize rewards
    total_rewards = np.zeros((num_episodes,))

    # loop for each episode
    for i in range(num_episodes):

        # initialize state
        s, info = env.reset()
        total_reward = 0

        # loop for each step of episode
        while True:

            # choose action from state using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                # choose the action with highest Q(s,a), if multiple, choose randomly
                values_ = Q[s, :]
                a = np.random.choice([action_ for action_, value_ in enumerate(
                    values_) if value_ == np.max(values_)])

            # take action, observe reward and next state
            s_, r, terminated, truncated, info = env.step(a)

            # update Q
            Q[s, a] = Q[s, a] + alpha * \
                (r + gamma * np.max(Q[s_, :]) - Q[s, a])

            # update state
            s = s_

            # update total reward
            total_reward += r

            # until state is terminal
            if terminated:
                total_rewards[i] = total_reward
                break

    return Q, total_rewards

def sarsa(alpha, epsilon, gamma, num_episodes, env):
    # initialize Q(s,a)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # initialize rewards
    total_rewards = np.zeros((num_episodes,))

    # loop for each episode
    for i in range(num_episodes):

        # initialize state
        s, info = env.reset()
        total_reward = 0

        # choose action from state using policy derived from Q (e-greedy)
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            # choose the action with highest Q(s,a), if multiple, choose randomly
            values_ = Q[s, :]
            a = np.random.choice([action_ for action_, value_ in enumerate(
                values_) if value_ == np.max(values_)])

        # loop for each step of episode
        while True:

            # take action, observe reward and next state
            s_, r, terminated, truncated, info = env.step(a)

            # choose action from state using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                a_ = env.action_space.sample()
            else:
                # choose the action with highest Q(s,a), if multiple, choose randomly
                values_ = Q[s_, :]
                a_ = np.random.choice([action_ for action_, value_ in enumerate(
                    values_) if value_ == np.max(values_)])

            # update Q
            Q[s, a] = Q[s, a] + alpha * \
                (r + gamma * Q[s_, a_] - Q[s, a])

            # update state
            s = s_
            a = a_

            # update total reward
            total_reward += r

            # until state is terminal
            if terminated:
                total_rewards[i] = total_reward
                break

    return Q, total_rewards

if __name__ == '__main__':

    # Create the environment
    env = gym.make('CliffWalking-v0')

    # number of episodes
    num_episodes = 500

    # number of runs
    num_runs = 50

    # rewards history
    q_rewards_history = np.zeros((num_runs, num_episodes))
    sarsa_rewards_history = np.zeros((num_runs, num_episodes))

    # algorithm parameters
    alpha = 0.5
    epsilon = 0.1
    gamma = 1

    # loop for each run
    for r in range(num_runs):

        # run Q-learning algorithm
        q_q_learning, q_total_rewards = q_learning(alpha, epsilon, gamma, num_episodes, env)

        # run SARSA algorithm
        q_sarsa, sarsa_total_rewards = sarsa(alpha, epsilon, gamma, num_episodes, env)

        # store rewards
        q_rewards_history[r, :] = q_total_rewards
        sarsa_rewards_history[r, :] = sarsa_total_rewards

    # plot learning curve
    plt.figure()
    plt.plot(np.mean(q_rewards_history, axis=0), label='Q-learning')
    plt.plot(np.mean(sarsa_rewards_history, axis=0), label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()


    # print optimal policy, changes actions (0,1,2,3) to up, right, down, left
    optimal_policy = np.argmax(q_q_learning, axis=1)
    optimal_policy = np.reshape(optimal_policy, (4, 12))
    optimal_policy = np.array([['↑', '→', '↓', '←'][action] for action in optimal_policy.flatten()])
    optimal_policy = np.reshape(optimal_policy, (4, 12))
    print('Q-learning')
    print(optimal_policy)

    optimal_policy = np.argmax(q_sarsa, axis=1)
    optimal_policy = np.reshape(optimal_policy, (4, 12))
    optimal_policy = np.array([['↑', '→', '↓', '←'][action] for action in optimal_policy.flatten()])
    optimal_policy = np.reshape(optimal_policy, (4, 12))
    print('Sarsa')
    print(optimal_policy)
