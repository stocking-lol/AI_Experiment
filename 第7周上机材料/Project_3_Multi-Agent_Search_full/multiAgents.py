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
# Note: use it for educational purposes in School of Artificial Intelligence, Sun Yat-sen University. 
# Lecturer: Zhenhui Peng (pengzhh29@mail.sysu.edu.cn)
# Credit to UC Berkeley (http://ai.berkeley.edu)
# February, 2022


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function. 在每个决策点，通过一个状态评估函数分析它的可能行动来做决定

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.
        getAction根据评估函数选择最佳的行动
        getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        返回 {NORTH, SOUTH, WEST, EAST, STOP} 中的一个行动
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best 如果有多个最佳行动（分数相同），随机选一个

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
        # currentGameState、successorGameState的内部内容和函数可以查看pacman.py里的gameState类
        successorGameState = currentGameState.generatePacmanSuccessor(action) # 在当前状态后采取一个行动后到达的状态
        newPos = successorGameState.getPacmanPosition() # 下一个状态的位置 （x，y）
        newFood = successorGameState.getFood() # 下一个状态时环境中的食物情况 (TTTFFFFFT......)
        newGhostStates = successorGameState.getGhostStates() # 下一个状态时幽灵的状态
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # 吃了大的白色食物（能量点）后，白幽灵的剩余持续时间

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        currentFoodCount = currentGameState.getFood().count()
        newFoodCount = newFood.count()
        if (currentFoodCount > newFoodCount):
            score += 200

        foodlist = newFood.asList()
        if (foodlist):
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodlist)
            score += 15 / (minFoodDistance + 1)

        for ghost, scaretime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghost.getPosition()
            distance = manhattanDistance(newPos, ghostPos)

            if (scaretime > 0):
                score += 500 / (distance + 1)

            else:
                if (distance <= 1):
                    score -= 1000
                elif (distance <= 2):
                    score -= 500
                elif (distance <= 3):
                    score -= 200

        return score

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. 根据当前的游戏状态，返回一个根据minimax值选的最佳行动

        Here are some method calls that might be useful when implementing minimax.
        以下的一些函数调用可能会对你有帮助
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent 返回一个agent（包括吃豆人和幽灵）合法行动（如不能往墙的地方移动）的列表
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action 一个agent采取行动后，生成的新的游戏状态

        gameState.getNumAgents():
        Returns the total number of agents in the game 获取当前游戏中所有agent的数量

        gameState.isWin():
        Returns whether or not the game state is a winning state 判断一个游戏状态是不是目标的胜利状态

        gameState.isLose():
        Returns whether or not the game state is a losing state 判断一个游戏状态是不是游戏失败结束的状态
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = -float('inf')
            actions = state.getLegalActions(0)
            for action in actions:
                successor = state.generateSuccessor(0, action)
                value = max(value, min_value(successor, 1, depth))
            return value

        def min_value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = float('inf')
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    # Last ghost, next is Pacman's turn with depth-1
                    new_value = max_value(successor, depth - 1)
                else:
                    # Next ghost's turn
                    new_value = min_value(successor, agentIndex + 1, depth)
                value = min(value, new_value)
            return value

            # Main logic to choose best action

        best_score = -float('inf')
        best_action = Directions.STOP
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            score = min_value(successor, 1, self.depth)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            value = -float('inf')
            actions = state.getLegalActions(0)
            for action in actions:
                successor = state.generateSuccessor(0, action)
                value = max(value, min_value(successor, 1, depth, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = float('inf')
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                # Last ghost, next is Pacman's turn with depth-1
                    new_value = max_value(successor, depth - 1, alpha, beta)
                else:
                # Next ghost's turn
                    new_value = min_value(successor, agentIndex + 1, depth, alpha, beta)
                value = min(value, new_value)
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        # Main logic to choose best action

        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        best_action = Directions.STOP
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            score = min_value(successor, 1, self.depth, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            if best_score > beta:
                return best_action
            alpha = max(alpha, score)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(state, agentIndex, depth):
            # 终止条件：游戏结束或达到最大深度
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Pacman的Max节点
            if agentIndex == 0:
                maxVal = -float('inf')
                actions = state.getLegalActions(0)

                # 遍历所有合法动作
                for action in actions:
                    successor = state.generateSuccessor(0, action)
                    # 处理下一个agent（第一个幽灵）
                    maxVal = max(maxVal, value(successor, 1, depth))
                return maxVal

            # 幽灵的Expectation节点
            else:
                expVal = 0
                actions = state.getLegalActions(agentIndex)
                prob = 1.0 / len(actions)  # 均匀概率

                # 遍历所有幽灵动作
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)

                    # 计算下一个agent和深度
                    if agentIndex == numAgents - 1:
                        nextAgent = 0
                        nextDepth = depth - 1
                    else:
                        nextAgent = agentIndex + 1
                        nextDepth = depth

                    # 递归计算子节点值
                    expVal += prob * value(successor, nextAgent, nextDepth)
                return expVal

        # Main logic to choose best action

        maxValue = -float('inf')
        best_action = Directions.STOP
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            currentValue = value(successor, 1, self.depth)
            if currentValue > maxValue or (value == maxValue and best_action == Directions.STOP):
                maxValue = currentValue
                best_action = action
        return best_action
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodMatrix = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # 基础得分
    score = currentGameState.getScore()

    # 计算到最近食物的距离（反向激励）
    foodList = foodMatrix.asList()
    if foodList:
        minFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 4.0 / (minFoodDist + 1)  # 距离越近奖励越高

    # 剩余食物惩罚（越少越好）
    score -= 2.0 * len(foodList)

    # 处理幽灵状态
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:  # 幽灵可被吃状态
            if distance < ghost.scaredTimer:  # 在恐惧时间内可追击
                score += 150.0 / (distance + 1) + ghost.scaredTimer * 2
        else:  # 正常幽灵状态
            if distance <= 1:
                score -= 1500  # 致命危险
            elif distance <= 3:
                score -= 100.0 / (distance + 1)

    # 能量豆策略（鼓励收集胶囊）
    score -= 20 * len(capsules)  # 剩余胶囊越少越好

    # 激活胶囊奖励（当有幽灵处于恐惧状态时）
    activeScare = sum([g.scaredTimer for g in ghostStates])
    score += activeScare * 10  # 恐惧时间越长奖励越高

    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
