from gym.envs.registration import register as gym_register


env_list = []


def register(id, *args, **kwargs):
    assert id.startswith('DPOMDP-')
    assert id not in env_list
    env_list.append(id)
    gym_register(id, *args, **kwargs)