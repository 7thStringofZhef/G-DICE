import unittest

import gym
import gym_dpomdps


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
        actions = []
        actions = [(i,j) for i in range(env.action_space[0].n) for j in range(env.action_space[1].n)]*10
        #actions = [list(range(env.action_space[0].n)), list(range(env.action_space[1].n))] * 10

        env.seed(seed)
        env.reset()
        outputs = list(map(env.step, actions))

        env.seed(seed)
        env.reset()
        outputs2 = list(map(env.step, actions))
        self.assertEqual(outputs, outputs2)