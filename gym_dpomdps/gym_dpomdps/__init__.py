from pkg_resources import resource_listdir, resource_filename, resource_isdir
from .envs import DPOMDP
from .wrappers import MultiDPOMDP
from .envs.registration import register, env_list


def list_dpomdps():
    return list(env_list)


def is_dpomdp(name):
    return (name.upper().endswith('.DPOMDP') and
            not resource_isdir('gym_dpomdps.dpomdps', name))


fnames = filter(is_dpomdp, resource_listdir('gym_dpomdps.dpomdps', ''))
for fname in fnames:
    fpath = resource_filename('gym_dpomdps.dpomdps', fname)
    name = '.'.join(fname.split('.')[:-1])

    register(
        id=f'DPOMDP-{name}-v0',
        entry_point='gym_dpomdps.envs:DPOMDP',
        kwargs=dict(path=fpath, episodic=False),
    )

    register(
        id=f'DPOMDP-{name}-episodic-v0',
        entry_point='gym_dpomdps.envs:DPOMDP',
        kwargs=dict(path=fpath, episodic=True),
    )