import matplotlib.pyplot as plt
import numpy as np

from bandit import Bandit


EPSILONS = [0.001, 0.005, 0.01, 0.05]
N_BANDITS = 10
MEAN_RANGE = [-100, 100]
SD_RANGE = [10, 50]
ITERATIONS = 10_000


def main():
    bandits = init_bandits()
    players = [EpsilonGreedy(eps, bandits) for eps in EPSILONS]
    for _ in range(ITERATIONS):
        for player in players:
            player.play()
    rewards = [player.get_rewards() for player in players]
    plt.figure(figsize=[16, 8])
    plt.subplot(2, 1, 1)
    for reward, player in zip(rewards, players):
        mean_reward = reward[1]
        plt.plot(mean_reward, label=player.eps)
        plt.legend()
        plt.ylabel('Mean Reward (Cumulative)')
    plt.subplot(2, 1, 2)
    for reward in rewards:
        cum_reward = reward[0]
        plt.plot(cum_reward)
        plt.xlabel('No. Plays')
        plt.ylabel('Total Reward (Cumulative)')
    plt.show()
    
def init_bandits():
    means = np.random.uniform(*MEAN_RANGE, N_BANDITS)
    sds = np.random.uniform(*SD_RANGE, N_BANDITS)
    bandits = [Bandit(mu, sig) for mu, sig in zip(means, sds)]
    print('Expected Rewards:', [f'{x:.2f}' for x in means])
    return bandits


class EpsilonGreedy:
    def __init__(self, eps, bandits):
        self.eps = eps
        self.bandits = bandits
        self.n_bandits = len(bandits)
        self.bandit_rewards = [[] for _ in range(self.n_bandits)]
        self.total_rewards = []
        self.expected_rewards = {x: np.nan for x in range(self.n_bandits)}
        self.current_best_bandit = 0
        self.other_bandits = range(1, self.n_bandits)
        self.current_best_expected_reward = np.nan

    def play(self):
        n = np.random.random()
        choice = 'random' if n <= self.eps else 'best'
        bandit = (self.current_best_bandit if choice == 'best'
                  else np.random.choice(self.other_bandits, 1)[0])
        reward = self.bandits[bandit].play()
        self.total_rewards.append(reward)
        self._update_stats(bandit, reward)

    def _update_stats(self, bandit, reward):
        self.bandit_rewards[bandit].append(reward)
        self.expected_rewards[bandit] = (np.array(self.bandit_rewards[bandit])
                                         .mean())
        self._update_current_best()

    def _update_current_best(self):
        for bandit, expected_reward in self.expected_rewards.items():
            if expected_reward > self.current_best_expected_reward:
                self.current_best_expected_reward = expected_reward
                self.current_best_bandit = bandit
                self.other_bandits = (list(range(self.n_bandits))
                                      .remove(self.current_best_bandit))
                
    def get_rewards(self):
        cum_reward = np.cumsum(self.total_rewards)
        cum_mean = cum_reward / range(1, ITERATIONS + 1)
        return cum_reward, cum_mean

if __name__ == '__main__':
    main()
