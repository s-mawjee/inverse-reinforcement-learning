from gym.envs.registration import register

register(
    id='highway-law-v0',
    entry_point='environments.highway.envs.highway:Highway',
    kwargs={'reward': 10, 'collided_reward': -50, 'grass_reward': -10000}
)
register(
    id='highway-nasty-v0',
    entry_point='environments.highway.envs.highway:Highway',
    kwargs={'reward': 1, 'collided_reward': 10, 'grass_reward': -1}
)
register(
    id='highway-nice-v0',
    entry_point='environments.highway.envs.highway:Highway',
    kwargs={'reward': 1, 'collided_reward': -5, 'grass_reward': 0}
)
