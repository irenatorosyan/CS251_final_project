import gym
import numpy as np
import pygame as pygame

from functions import MonteCarloLearnStateValueFunction
from functions import evaluatePolicy
from functions import grid_print

# environment with only 1 hole to need less iterations for the showcase
desc = ["SFFF", "FFFF", "FFFF", "HFFG"]

# used for illustration purposes
# env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True,render_mode="human")

# used instead of the previous line to generate a large amount of iterations
env = gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)

stateNumber = env.observation_space.n
numberOfEpisodes = 10000
discountRate = 1

estimatedValuesMonteCarlo = MonteCarloLearnStateValueFunction(env, stateNumber=stateNumber, numberOfEpisodes=numberOfEpisodes, discountRate=discountRate)


# for comparison compute the state value function vector by using the iterative policy

# initial policy: completely random <- equal probability of choosing a particular action
initialPolicy = (1/4) * np.ones((16, 4))

valueFunctionVectorInitial = np.zeros(env.observation_space.n)
maxNumberOfIterationsOfIterativePolicyEvaluation = 1000
convergenceToleranceIterativePolicyEvaluation = 10 ** (- 6)

valueFunctionIterativePolicyEvaluation = evaluatePolicy(env, valueFunctionVectorInitial, initialPolicy, 1, maxNumberOfIterationsOfIterativePolicyEvaluation, convergenceToleranceIterativePolicyEvaluation)

grid_print(valueFunctionIterativePolicyEvaluation, reshapeDim=4, fileNameToSave='iterativePolicyEvaluationEstimated.png')
grid_print(estimatedValuesMonteCarlo, reshapeDim=4, fileNameToSave='monteCarloEstimated.png')
