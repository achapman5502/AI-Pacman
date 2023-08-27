# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


#archapm
# THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY

# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Alec Chapman



from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        minFood = float("inf")
        for food in newFoodList: 
            minFood = min(minFood, manhattanDistance(newPos, food)) 
        
        for ghost in successorGameState.getGhostPositions(): #if ghost is nearby, we avoid it
            if manhattanDistance(newPos, ghost) < 2:
                return float("-inf")
        
        return successorGameState.getScore() + 1 / minFood #return reciprocal

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex, depth): 
        if agentIndex >= gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #if depth is reached or game is won or lost
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth): 
        v = float("-inf") #v = -infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successors, agentIndex + 1, depth)
            v = max(v, value)
        return v
    
    def minValue(self, gameState, agentIndex, depth): 
        v = float("inf") #v = infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successors, agentIndex + 1, depth)
            v = min(v, value)
        return v

    def getAction(self, gameState):
        agentIndex = 0
        maximum = float("-inf") #initialize maximum to -infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            v = self.minimax(successors, 1, 0)
            if v > maximum:
                maximum = v
                bestAction = action
        return bestAction
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def prune(self, gameState, agentIndex, depth, alpha, beta): 
        if agentIndex >= gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #if depth is reached or game is won or lost
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta): 
        v = float("-inf") #v = -infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.prune(successors, agentIndex + 1, depth, alpha, beta)
            v = max(v, value)
            if v > beta:
              return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        v = float("inf") #v = infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.prune(successors, agentIndex + 1, depth, alpha, beta)
            v = min(v, value)
            if v < alpha:
              return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        agentIndex = 0
        alpha = float("-inf") #initialize alpha to -infinity and beta to infinity
        beta = float("inf")
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            v = self.prune(successors, 1, 0, alpha, beta)
            if v > alpha:
                alpha = v
                bestAction = action
        return bestAction

    


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, agentIndex, depth): 
        if agentIndex >= gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #if depth is reached or game is won or lost
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expectedValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth): 
        v = float("-inf") #v = -infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.expectimax(successors, agentIndex + 1, depth)
            v = max(v, value)
        return v
    
    def expectedValue(self, gameState, agentIndex, depth): 
        v = 0 #v = 0
        actions = gameState.getLegalActions(agentIndex)
        p = 1.0 / len(actions)
        for action in actions: #for each successor of state
            successors = gameState.generateSuccessor(agentIndex, action)
            value = self.expectimax(successors, agentIndex + 1, depth)
            v = v + p * value
        return v

    def getAction(self, gameState):
        agentIndex = 0
        maximum = float("-inf") #initialize max to -infinity
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            v = self.expectimax(successors, 1, 0)
            if v > maximum:
                maximum = v
                bestAction = action
        return bestAction
            


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFoodList = currentGameState.getFood().asList()
    minFood = float("inf")
    for food in newFoodList:
        minFood = min(minFood, manhattanDistance(newPos, food))
    return currentGameState.getScore() + 1 / minFood

# Abbreviation
better = betterEvaluationFunction
