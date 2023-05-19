def MonteCarloLearnStateValueFunction(env, stateNumber, numberOfEpisodes, discountRate):
    # learns the state value function
    # env - OpenAI Gym environment
    # output - final learned value of the state value function vector

    import numpy as np
    
    sumReturnForEveryState = np.zeros(stateNumber)
    numberVisitsForEveryState = np.zeros(stateNumber)
    valueFunctionEstimate = np.zeros(stateNumber)
    
    # episode simulation
    for indexEpisode in range(numberOfEpisodes):
        visitedStatesInEpisode = []
        rewardInVisitedState = []
        (currentState, prob) = env.reset()
        visitedStatesInEpisode.append(currentState)
        
        print("Simulating episode {}".format(indexEpisode))

        # single episode simulation
        # generate actions -> step according to them
        # terminal state -> break
        while True:
            randomAction = env.action_space.sample()
            # accepts: action
            # returns: observed state, reward that is the result of taking the action, is it a terminal state,
            #          transition probability

            (currentState, currentReward, terminalState, _, _) = env.step(randomAction)
            
            rewardInVisitedState.append(currentReward)
            
            if not terminalState:
                visitedStatesInEpisode.append(currentState)   
            else:
                break
            # terminal state is not included in the list of visited
        
        numberOfVisitedStates = len(visitedStatesInEpisode)
            
        Gt: int = 0
        for indexCurrentState in range(numberOfVisitedStates-1, -1, -1):
                
            stateTmp = visitedStatesInEpisode[indexCurrentState]
            returnTmp = rewardInVisitedState[indexCurrentState]
            # summing the returns backwards
              
            Gt = discountRate * Gt + returnTmp
              
            if stateTmp not in visitedStatesInEpisode[0: indexCurrentState]:
                numberVisitsForEveryState[stateTmp] = numberVisitsForEveryState[stateTmp] + 1
                sumReturnForEveryState[stateTmp] = sumReturnForEveryState[stateTmp] + Gt
            
    # final estimate of the state value function vector
    for indexSum in range(stateNumber):
        if numberVisitsForEveryState[indexSum] != 0:
            valueFunctionEstimate[indexSum] = sumReturnForEveryState[indexSum] / numberVisitsForEveryState[indexSum]
        
    return valueFunctionEstimate


def evaluatePolicy(env, valueFunctionVector, policy, discountRate, maxNumberOfIterations, convergenceTolerance):
    # used for comparison - uses iterative policy evaluation algorithm to compute the state value function
    # valueFunctionVector - initial state value function vector
    # policy - policy to be evaluated - this is a matrix with the dimensions (number of states)x(number of actions)
    #        - p,q entry of this matrix is the probability of selection action q in state p
    # discountRate - discount rate
    # outputs: final value of the state value function vector
    import numpy as np
    convergenceTrack = []
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector, 2))
        valueFunctionVectorNextIteration = np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum = 0
            for action in env.P[state]:
                innerSum = 0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    innerSum = innerSum + probability * (reward+discountRate * valueFunctionVector[nextState])
                outerSum = outerSum + policy[state, action] * innerSum
            valueFunctionVectorNextIteration[state] = outerSum
        if np.max(np.abs(valueFunctionVectorNextIteration - valueFunctionVector)) < convergenceTolerance:
            valueFunctionVector = valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunctionVector = valueFunctionVectorNextIteration
    return valueFunctionVector


def grid_print(valueFunction, reshapeDim, fileNameToSave):
    # visualizes and saves the state value function
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.heatmap(valueFunction.reshape(reshapeDim,reshapeDim),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig(fileNameToSave, dpi=600)
    plt.show()
        