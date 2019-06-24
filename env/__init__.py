from gym.envs.registration import register


########################################################################################
# REGRESSION
register(
    id='Regression-v0',
    entry_point='env.regression.regression_env:RegressionEnv',
    max_episode_steps=100
)
