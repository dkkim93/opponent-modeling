import gym
import scipy.stats
import numpy as np
from gym import spaces


class RegressionEnv(gym.Env):
    """Regression task based on Gausssian
    - Observation: sample uniformly sampled between -x and x
    - Action: Equal to observation, so action is not used in here
    - Reward: Gaussian PDF according to the sampled sample
    """
    def __init__(self, n_agent):
        super(RegressionEnv, self).__init__()
        self.n_agent = n_agent
        self.observation_space = spaces.Box(low=-10., high=10., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        assert n_agent <= 2, "Up to two agents are supported"

    def step(self, actions):
        next_observations, rewards, done, info = [], [], False, {}
        for i_agent in range(self.n_agent):
            next_observations.append(self.observation_space.sample()[0])
            rewards.append(self.gauss_dists[i_agent].pdf(actions[i_agent]))

        return (next_observations, rewards, done, info)

    def reset(self):
        samples = []
        for i_agent in range(self.n_agent):
            samples.append(self.observation_space.sample()[0])

        return samples

    def render(self):
        raise NotImplementedError("")

    def reset_task(self, task):
        means = []
        for i_agent in range(self.n_agent):
            if i_agent == 0:
                std = 2
                means.append(self.observation_space.low[0] + std * 2 + 0.1 * task)
            elif i_agent == 1:
                std = 1
                means.append(self.observation_space.low[0] + std * 10 + 0.1 * task)
            else:
                raise ValueError("Invalid n_agent")

        self.gauss_dists = []
        for i_agent in range(self.n_agent):
            std = 2 if i_agent == 0 else 1
            self.gauss_dists.append(scipy.stats.norm(loc=means[i_agent], scale=std))
