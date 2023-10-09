import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)


register(
    id='AlienGoEnv-v0',
    entry_point='envs.AlienGoGymEnv:AlienGoGymEnv',
    max_episode_steps=1000,
    reward_threshold=100.0,
)

register(
    id='QuadrupedGymEnv-v0',
    entry_point='envs.pmtg_task:QuadrupedGymEnv',
    max_episode_steps=2000,
    reward_threshold=100.0,
)