import numpy as np


# Evaluate a single sample, starting from first node
# Inputs:
#   env: Environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSamplePOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions, numSimulations=1):
    gamma = env.discount if env.discount is not None else 1
    values = np.zeros(numSimulations, dtype=np.float64)
    for sim in range(numSimulations):
        env.reset()
        currentNodeIndex = 0
        currentTimestep = 0
        isDone = False
        value = 0.0
        while not isDone and currentTimestep < timeHorizon:
            obs, reward, isDone = env.step(actionTransitions[currentNodeIndex])[:3]
            currentNodeIndex = nodeObservationTransitions[obs, currentNodeIndex]
            value += reward * (gamma ** currentTimestep)
            currentTimestep += 1
        values[sim] = value
    return values.mean(), values.std()

# Evaluate a single sample, starting from first node
# Inputs:
#   env: DPOMDP environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes, ) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSampleDPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions, numSimulations=1):
    nAgents = env.agents
    agentIndices = np.arange(nAgents, dtype=int)
    gamma = env.discount if env.discount is not None else 1
    values = np.zeros(numSimulations, dtype=np.float64)
    for sim in range(numSimulations):
        env.reset()
        currentNodeIndex = [0 for _ in range(nAgents)]
        currentTimestep = 0
        isDone = False
        value = 0.0
        while not isDone and currentTimestep < timeHorizon:
            obs, reward, isDone = env.step(actionTransitions[currentNodeIndex, agentIndices])[:3]
            currentNodeIndex = nodeObservationTransitions[obs, currentNodeIndex, agentIndices]
            value += reward * (gamma ** currentTimestep)
            currentTimestep += 1
        values[sim] = value
    return values.mean(), values.std()

# Evaluate multiple trajectories for a sample, starting from first node
# Inputs:
#   env: MultiPOMDP environment in which to evaluate (numTrajectories is number of simulations)
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSampleMultiPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions):
    numTrajectories = env.nTrajectories
    gamma = env.discount if env.discount is not None else 1
    env.reset()
    currentNodes = np.zeros(numTrajectories, dtype=np.int32)
    currentTimestep = 0
    values = np.zeros(numTrajectories, dtype=np.float64)
    isDones = np.zeros(numTrajectories, dtype=bool)
    while not all(isDones) and currentTimestep < timeHorizon:
        obs, rewards, isDones = env.step(actionTransitions[currentNodes])[:3]
        currentNodes = nodeObservationTransitions[obs, currentNodes]
        values += rewards * (gamma ** currentTimestep)
        currentTimestep += 1

    return values.mean(axis=0), values.std(axis=0)

# Evaluate multiple trajectories for a sample, starting from first node
# Inputs:
#   env: MultiDPOMDP environment in which to evaluate (numTrajectories is number of simulations)
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes, numAgents) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes, numAgents) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSampleMultiDPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions):
    nTrajectories = env.nTrajectories
    nAgents = env.agents
    agentIndices = tuple(np.full(nTrajectories, a, dtype=np.int32) for a in range(nAgents))
    gamma = env.discount if env.discount is not None else 1
    env.reset()
    currentNodes = tuple(np.zeros(nTrajectories, dtype=np.int32) for _ in range(nAgents))
    currentTimestep = 0
    values = np.zeros(nTrajectories, dtype=np.float64)
    isDones = np.zeros(nTrajectories, dtype=bool)
    while not all(isDones) and currentTimestep < timeHorizon:
        obs, rewards, isDones = env.step(actionTransitions[currentNodes, agentIndices].T)[:3]
        currentNodes = nodeObservationTransitions[tuple(obs[:, i] for i in range(nAgents)), currentNodes, agentIndices]
        values += rewards * (gamma ** currentTimestep)
        currentTimestep += 1

    return values.mean(axis=0), values.std(axis=0)

def runDeterministicControllerOnEnvironment(env, controller, timeHorizon, printMsgs=False):
    gamma = env.discount if env.discount is not None else 1
    env.reset(printEnv=print)
    currentTimestep = 0
    isDone = False
    value = 0.0
    while not isDone and currentTimestep < timeHorizon:
        action = controller.getAction()
        if printMsgs:
            print('Timestep', currentTimestep)
            print('action', action)
        obs, reward, isDone = env.step(action, printEnv=printMsgs)[:3]
        print('reward:', reward)
        controller.processObservation(obs)
        value += reward * (gamma ** currentTimestep)
        currentTimestep += 1
    return value