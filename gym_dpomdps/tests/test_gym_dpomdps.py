import unittest

import gym
import gym_dpomdps

import numpy as np


class GymDPOMDPs_Test(unittest.TestCase):
    def test_list_dpomdps(self):
        dpomdps = gym_dpomdps.list_dpomdps()

        self.assertTrue(len(dpomdps) > 0)
        for dpomdp in dpomdps:
            self.assertTrue(dpomdp.startswith('DPOMDP'))

    @unittest.skip
    def test_make(self):
        envs = {}
        for dpomdp in gym_dpomdps.list_pomdps():
            try:
                env = gym.make(dpomdp)
            except MemoryError:  # some POMDPs are too big
                pass
            else:
                envs[dpomdp] = env

    def test_seed(self):
        env = gym.make('DPOMDP-dectiger-v0')

        seed = 17
        actions = [(i,j) for i in range(env.action_space[0].n) for j in range(env.action_space[1].n)]*10

        env.seed(seed)
        env.reset()
        outputs = list(map(env.step, actions))

        env.seed(seed)
        env.reset()
        outputs2 = list(map(env.step, actions))
        self.assertEqual(outputs, outputs2)

    def test_multi(self):
        env = gym.make('DPOMDP-recycling-v0')
        multiEnv = gym_dpomdps.MultiDPOMDP(env, 50)
        for timestep in range(50):
            agent1Actions = np.random.choice(np.arange(env.action_space[0].n, dtype=int), 50)
            agent2Actions = np.random.choice(np.arange(env.action_space[1].n, dtype=int), 50)
            actions = np.stack((agent1Actions, agent2Actions), axis=1)
            obs, rewards, done, _ = multiEnv.step(actions)