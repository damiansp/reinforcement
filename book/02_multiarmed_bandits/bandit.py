import numpy as np


class Bandit:
    def __init__(self, mean_reward, sd):
        self._mean = mean_reward
        self._sd = sd

    def play(self):
        return np.random.normal(self._mean, self._sd)

    def __str__(self):
        return f'Bandit with mean reward: {self._mean} and sd: {self._sd}'

    @property
    def mean(self):
        return self._mean



if __name__ == '__main__':
    mu = 3
    sd = 5
    plays = 100
    bandit = Bandit(mu, sd)
    print(bandit)
    payoffs = np.array([bandit.play() for _ in range(plays)])
    est = payoffs.mean()
    print(f'Estimated Reward: {est:.4f}')
