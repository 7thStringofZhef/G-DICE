from gym_pomdps import POMDP


from enum import Enum

import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiBinary
from gym.utils import seeding

from functools import partial
from itertools import chain

#from gym_pomdp.envs.coord import Grid, Coord
#from gym_pomdp.envs.gui import ShipGui

# Battleship
# 1 agent
# Discount: 1
# Obs: 0 (miss), 1 (hit)
# Actions: 100 (10x10 firing)
#   Not preferred: Remove taken actions
#   Preferred: Remove useless actions as well (diagonal from hits)
# State of each cell: 0 (empty), 1 (empty, visited), 2 (empty, diagonal), 3 (occupied), 4 (occupied, visited),

class BattleshipPOMDP(POMDP):
    def __init__(self, boardSize=10, seed=None, prune=True):
        self.seed(seed)
        self.prune = prune
        self.boardSize = boardSize
        self.observation_space = Discrete(2)  # Miss or hit
        self.action_space = Discrete(boardSize ** 2)  # Fire on any square
        self.reward_range = -1, 100  # Range of rewards. -1 for each timestep, 100 for winning
        self.actionMap = partial(np.unravel_index, dims=(boardSize, boardSize))  # Function to get flat indices from tuple
        self.actionUnMap = partial(np.ravel_multi_index, dims=(boardSize, boardSize))  # Function to get tuple indices from flat
        self.reset()

    def reset(self):
        # Occupied, visited, diagonal
        self.occupied = np.zeros(self.boardSize ** 2, dtype=bool)
        self.visited = np.zeros(self.boardSize ** 2, dtype=bool)
        self.diagonal = np.zeros(self.boardSize ** 2, dtype=bool)

        self.legalMoves = np.ones(self.boardSize ** 2, dtype=bool)  # Which squares have been fired on? Mark those as false
        self.preferredMoves = np.ones(self.boardSize ** 2, dtype=bool)  # Additionally prune squares diagonal to hit
        self._initShips()

    # Place a valid configuration of ships on the board
    def _initShips(self):
        self.invCheck = lambda pos: self.occupied[pos] or self.diagonal[pos]  # Check square occupied/adjacency status
        self.shipLengths = [5, 4, 3, 3, 2]  # Carrier, battleship, cruiser, sub, destroyer
        self.numSegments = np.sum(self.shipLengths)
        positions = np.arange(self.boardSize ** 2)  # Flat indices 0-100
        directions = np.arange(4)  # Directions
        permutations = np.array(np.meshgrid(positions, directions)).reshape(2, -1).swapaxes(0, 1)
        shuffleInd = self.np_random.permutation(directions.size * positions.size)  # Shuffle permutations
        for ship in self.shipLengths:
            perm = permutations[shuffleInd, :]
            for pos, dir in perm:
                valid = self._valid(pos, dir, ship)
                if valid:
                    self._placeShip(valid)
                    break

    # Does placing this new ship collide with previous ships? Or with the grid?
    # shipPos is flat index
    # shipDir is 0 (N), 1 (E), 2 (S), 3 (W)
    def _valid(self, shipPos, shipDir, shipLength):
        if self.invCheck(shipPos): return []
        shipPosAsTuple = self.actionMap(shipPos)
        allPos = [shipPosAsTuple]
        # Get all positions (reverse so I run into out-of-bounds errors faster)
        for i in reversed(range(1, shipLength)):
            if shipDir == 0:  # North
                newTup = (shipPosAsTuple[0] - i, shipPosAsTuple[1])
                if newTup[0] < 0 or self.invCheck(self.actionUnMap(newTup)): return []
                else: allPos.append(newTup)
            elif shipDir == 1:  # East
                newTup = (shipPosAsTuple[0], shipPosAsTuple[1] + i)
                if newTup[1] >= self.boardSize or self.invCheck(self.actionUnMap(newTup)): return []
                else: allPos.append(newTup)
            elif shipDir == 2:  # South
                newTup = (shipPosAsTuple[0] + i, shipPosAsTuple[1])
                if newTup[0] >= self.boardSize or self.invCheck(self.actionUnMap(newTup)): return []
                else: allPos.append(newTup)
            elif shipDir == 3:  # West
                newTup = (shipPosAsTuple[0], shipPosAsTuple[1] - i)
                if newTup[1] < 0 or self.invCheck(self.actionUnMap(newTup)): return []
                else: allPos.append(newTup)
        return allPos

    # Place ship according to a set of tuple coordinates
    def _placeShip(self, validPosition):
        self.occupied[[self.actionUnMap(p) for p in validPosition]] = True
        diagonalPos = list(set(chain.from_iterable([self._diagOfPos(valPos) for valPos in validPosition])))  # Convert to a set to make unique
        # Remove invalid indices
        diagonalPos = [pos for pos in diagonalPos if 0 <= pos[0] < self.boardSize and 0 <= pos[1] < self.boardSize]
        self.diagonal[[self.actionUnMap(p) for p in diagonalPos]] = True

    def _diagOfPos(self, valPos):
        return [(valPos[0] - 1, valPos[1] - 1), (valPos[0] - 1, valPos[1] + 1), (valPos[0] + 1, valPos[1] + 1), (valPos[0] + 1, valPos[1] - 1)]

    # Fire a shot
    def step(self, action):
        refire = False
        # Visit the square
        if self.occupied[action] and not self.visited[action]:
            obs = 1
        else:
            if self.visited[action]: refire = True  # Are we shooting the same square again?
            obs = 0
        self.visited[action] = True
        r = self._rewardFn()
        isDone = self._isDone()

        # Update set of legal moves
        if self.prune and not refire:
            self.legalMoves[action] = False
            self.preferredMoves[action] = False
            if obs:  # If a hit, can't be any of the diagonals
                actionAsTup = self.actionMap(action)
                diagonalPos = self._diagOfPos(actionAsTup)
                diagonalPos = [pos for pos in diagonalPos if 0 <= pos[0] < self.boardSize and 0 <= pos[1] < self.boardSize]
                self.preferredMoves[[self.actionUnMap(p) for p in diagonalPos]] = False
        return obs, r, isDone

    # +100 if you win. -1 for everything else
    def _rewardFn(self):
        return 100 if np.sum(np.logical_and(self.visited, self.occupied)) == self.numSegments else -1

    # Done if you win
    def _isDone(self):
        return np.sum(np.logical_and(self.visited, self.occupied)) == self.numSegments


# Grid flags: 1 is passable, 3 is food, 7 is power (0, 01, 11, 111 were bit flags)
class PocmanPOMDP(POMDP):
    # Flags
    PASS = 1
    SEED = 3
    POWER = 7

    # Static variables
    PassageY = -1  # The row (column?) at which an agent can wrap around to the other side of the maze
    SmellRange = 1  # How far can POCMAN smell food (adjacent, even diagonally)
    HearRange = 2  # How far can POCMAN hear ghosts (manhattan)
    FoodProb = 0.5  # How likely is it that a seed becomes food
    ChaseProb = 0.75  # How likely is it that ghost moves towards POCMAN when trying to attack
    DefensiveSlip = 0.25  # How often do ghosts fail to move (slip) when trying to escape
    PowerNumSteps = 15  # powerup duration

    # Rewards
    RewardClearLevel = 1000
    RewardDefault = -1
    RewardDie = -100
    RewardEatFood = 10
    RewardEatGhost = 25
    RewardHitWall = -25
    # If size
    def __init__(self, size='standard', seed=None):
        self.gamma = 0.95
        self.size = size
        self.seed(seed)
        self.reward_range = -100, 1000
        self.observation_space = MultiBinary(10)  # 10 bit binary observations
        self.action_space = Discrete(4)

    # Initialize standard 17*19 grid
    # 4 ghosts
    # Ghosts can sense at 6 manhattan distance
    def _initStandard(self):
        #** Transpose?
        self.grid = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3],
        [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
        [3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3],
        [3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3],
        [0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 0, 0, 0],
        [1, 1, 1, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 1, 1, 1],
        [0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3],
        [7, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 7],
        [0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0],
        [3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3],
        [3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], dtype=int)
        self.nGhosts = 4
        self.ghostRange = 6
        self.pocmanHome = (8, 6)
        self.ghostHome = (8, 10)
        self.passageY = 10

    # Initialize mini 10*10 grid
    #
    def _initMini(self):
        self.grid = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [3, 0, 3, 3, 3, 3, 3, 3, 0, 3],
        [3, 3, 3, 0, 0, 0, 0, 3, 3, 3],
        [0, 0, 3, 0, 1, 1, 3, 3, 0, 0],
        [0, 0, 3, 0, 1, 1, 3, 3, 0, 0],
        [3, 3, 3, 0, 0, 0, 0, 3, 3, 3],
        [3, 0, 3, 3, 3, 3, 3, 3, 0, 3],
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], dtype=int)
        self.foodGrid = np.logical_and(self.grid == self.SEED, self.grid == )
        self.nGhosts = 3
        self.ghostRange = 4
        self.pocmanHome = (4, 2)
        self.ghostHome = (4, 4)
        self.passageY = 5

    # Initialize micro 7*7 grid
    #
    def _initMicro(self):
        self.grid = np.array([[3, 3, 3, 3, 3, 3, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 3, 0, 3, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 3, 3, 3, 3, 3, 3]], dtype=int)
        self.nGhosts = 1
        self.ghostRange = 3
        self.pocmanHome = (3, 0)
        self.ghostHome = (3, 4)
        self.passageY = None

    def reset(self):
        if self.size is 'standard':
            self._initStandard()
        elif self.size is 'mini':
            self._initMini()
        elif self.size is 'micro':
            self._initMicro()

        self.numFood = np.sum(self.grid == self.POWER)

    def step(self, action):
        # Order of operations
        #   Start reward at default
        #   Move pocman (add reward if hit wall)
        #   Decrement power steps
        #   Check for hit ghost (send home if eaten)
        #   Move ghosts
        #   Check again
        #   Get observation
        #   Eat food
        #   Check if level cleared
        pass

    def _reward(self):
        pass

# Get the manhattan distance between two coordinates
def _manhattanDistance(tup1, tup2):
    return abs(tup1[0]-tup2[0]) + abs(tup1[1]-tup2[1])


if __name__ == "__main__":
    test = BattleshipPOMDP()
    test2 = test.step(0)
    pass