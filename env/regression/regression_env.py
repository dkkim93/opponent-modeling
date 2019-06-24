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
    def __init__(self):
        super(RegressionEnv, self).__init__()
        self.observation_space = spaces.Box(low=-10., high=10., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)

    def step(self, action):
        next_observation = self.observation_space.sample()[0]
        reward = self.gauss_dist.pdf(action)
        done = False
        info = {}

        return (next_observation, reward, done, info)

    def reset(self):
        sample = self.observation_space.sample()[0]
        return sample

    def render(self):
        raise NotImplementedError("")

    def reset_task(self, task):
        std = 2
        mean = self.observation_space.low[0] + std * 2 + 0.1 * task
        self.gauss_dist = scipy.stats.norm(loc=mean, scale=std)
