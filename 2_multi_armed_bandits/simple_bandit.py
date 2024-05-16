import numpy as np
import matplotlib.pyplot as plt

class bandit_algorithm:
    def __init__(self, bandit, epsilon, steps):
        self.bandit = bandit
        self.epsilon = epsilon
        self.steps = steps
        self.k = bandit.k
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.reward = np.zeros(self.steps)

    def learn(self):
        for t in range(self.steps):
            # epsilon greedy
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.k)
            else:
                # choose action with maximum Q, if multiple, choose randomly
                action = np.random.choice(np.where(self.Q == np.max(self.Q))[0])
            # get reward
            reward = self.bandit.bandit(action)
            # update Q
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
            # update reward
            self.reward[t] = reward

class bandit:
    def __init__(self, k):
        self.k = k
        self.mu = np.random.randn(k)
        self.sigma = np.ones(k)

    def bandit(self, action):
        return np.random.normal(self.mu[action], self.sigma[action])
    
if __name__ == '__main__':
    
    # set random seed for reproducibility
    np.random.seed(0)

    k = 10
    steps = 1000
    epsilon_list = [0, 0.1, 0.01]

    # mean reward
    number_of_runs = 2000
    rewards = np.zeros((len(epsilon_list), number_of_runs, steps))

    for i, epsilon in enumerate(epsilon_list):
        for j in range(number_of_runs):
            bandit_instance = bandit(k)
            simple_bandit = bandit_algorithm(bandit_instance, epsilon, steps)
            simple_bandit.learn()
            rewards[i, j, :] = simple_bandit.reward

    # plot
    plt.figure(figsize=(10, 6))
    for i in range(len(epsilon_list)):
        plt.plot(
            np.mean(rewards[i, :, :], axis=0),
            label="epsilon = {}".format(epsilon_list[i]),
        )
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.title("Average Reward vs Steps", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig("simple_bandit.svg")