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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestact = getact(self.minmax(gameState, self.index, 0))
        return bestact

    def minmax(self, state, index, depth):
        betteract = None
        if depth == self.depth or len(state.getLegalActions(index)) == 0:
            return [self.evaluationFunction(state), betteract]

        # Pacman is moving
        if index == 0:
            value = -float('inf')
            Actions = state.getLegalActions(index)
            for action in Actions:
                val = getval(self.minmax(state.generateSuccessor(index, action), 1, depth))
                if val > value:
                    value = val
                    betteract = action

        # ghosts are moving
        else:
            value = float('inf')
            Actions = state.getLegalActions(index)
            for action in Actions:
                if index == state.getNumAgents() - 1:
                    val = getval(self.minmax(state.generateSuccessor(index, action), self.index, depth + 1))
                else:
                    val = getval(self.minmax(state.generateSuccessor(index, action), index + 1, depth))
                if val < value:
                    value = val
                    betteract = action

        return [value, betteract]


def getval(link):
    return link[0]

def getact(link):
    return link[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestact = getact(self.alphabeta(gameState, self.index, 0))
        return bestact

    def alphabeta(self, state, index, depth, alpha=-float('inf'), beta=float('inf')):
        betteract = None
        if depth == self.depth or len(state.getLegalActions(index)) == 0:
            return [self.evaluationFunction(state), betteract]

        Actions = state.getLegalActions(index)
        value = float('inf')
        # Pacman is moving
        if index == self.index:
            value = -float('inf')
            for action in Actions:
                val = getval(self.alphabeta(state.generateSuccessor(index, action), index + 1, depth, alpha, beta))
                # 剪
                if val > alpha:
                    alpha = val
                if val > beta:
                    return [val, action]
                
                if val > value:
                    value = val
                    betteract = action

                    # ghosts are moving
        else:
            for action in Actions:
                if index == state.getNumAgents() - 1:
                    val = getval(
                        self.alphabeta(state.generateSuccessor(index, action), self.index, depth + 1, alpha, beta))
                else:
                    val = getval(self.alphabeta(state.generateSuccessor(index, action), index + 1, depth, alpha, beta))
                # 剪枝
                if val < beta:
                    beta = val
                if val < alpha:
                    return [val, action]
                
                if val < value:
                    value = val
                    betteract = action

        return [value, betteract]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestact = getact(self.expectimax(gameState, self.index, 0))
        return bestact

    def expectimax(self, state: GameState, index, depth):
        bestact = None
        if depth == self.depth or len(state.getLegalActions(index)) == 0:
            return [self.evaluationFunction(state), bestact]

        Actions = state.getLegalActions(index)
        # Pacman is moving
        if index == self.index:
            value = -float('inf')
            for action in Actions:
                val = getval(self.expectimax(state.generateSuccessor(index, action), index + 1, depth))
                if val > value:
                    value = val
                    bestact = action
            return [value, bestact]

        # ghosts are moving
        else:
            value = []
            for action in Actions:
                if index == state.getNumAgents() - 1:
                    val = getval(self.expectimax(state.generateSuccessor(index, action), self.index, depth + 1))
                else:
                    val = getval(self.expectimax(state.generateSuccessor(index, action), index + 1, depth))
                value.append(val)
            # 求期望值
            expectval = 0
            for i in value:
                expectval += i
            expectval = expectval / len(value)
            return [expectval, None]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    pacman = currentGameState.getPacmanPosition()
    score = 0

    food_mark = 10.0
    capsule_mark = 10.0
    ghost_mark = -5.0
    scaredghost_mark = 20.0

    # food
    distance_food = []
    for food in currentGameState.getFood().asList():
        distance_food.append(util.manhattanDistance(pacman, food))

    for lenth in distance_food:
        if lenth == 0:
            score += food_mark
        elif lenth <= 5:
            score += food_mark / lenth

    # capsule
    distance_capsule = []
    for capsule in currentGameState.getCapsules():
        distance_capsule.append(util.manhattanDistance(pacman, capsule))

    for lenth in distance_capsule:
        if lenth == 0:
            score += capsule_mark
        else:
            score += capsule_mark / lenth

    # ghost
    for ghost in currentGameState.getGhostStates():
        distance_ghost = util.manhattanDistance(pacman, ghost.getPosition())
        # whether the ghost is scared
        if ghost.scaredTimer > 0:
            score += scaredghost_mark / distance_ghost
        else:
            if distance_ghost == 0:
                return -float('inf')
            else:
                score += ghost_mark / distance_ghost

    return score + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
