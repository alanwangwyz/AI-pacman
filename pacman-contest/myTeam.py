# myTeam.py
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

from captureAgents import CaptureAgent

import random, util, sys, operator
from game import Directions
import copy
from util import nearestPoint
from game import Actions

arguments = {}
beliefs = {}
beliefsInitialized = []
minPacmanPos = {}
minGhostPos = {}

def createTeam(firstIndex, secondIndex, isRed,
               first='TopAgent', second='BottomAgent', **args):
    if 'numTraining' in args:
        arguments['numTraining'] = args['numTraining']
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ApproximateQAgent(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.numTraining = 0
        self.episodesCur = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2

    def registerInitialState(self, gameState):
        self.lastAction = None
        CaptureAgent.registerInitialState(self, gameState)
        self.legalAction, self.corners = self.initialLegalAction(gameState)

    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentPosition(self.index)
        if pos != nearestPoint(pos):

            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def isHomeSide(self, node, gameState):
        width, height = gameState.data.layout.width, gameState.data.layout.height
        if self.index in [0, 2]:
            if node[0] <width / 2-1:
                return True
            else:
                return False
        else:
            if node[0] >width / 2:
                return True
            else:
                return False

    def initialLegalAction(self, gameState):
        legalAction = {}
        corners = []
        walls = copy.deepcopy(gameState.getWalls())
        for x in range(0, gameState.data.layout.width):
            for y in range(0, gameState.data.layout.height):
                if not gameState.hasWall(x, y):
                    possible = []
                    adjacentToConers = 0
                    for action, position in Actions._directionsAsList:
                        px, py = position
                        ny = py + y
                        nx = px + x
                        if not walls[nx][ny]: possible.append(((nx, ny), action, 1))
                    legalAction[(x, y)] = possible
                    if len(possible) <= 2:
                        corners.append((x, y))
        return legalAction, corners

    def aStarSearch(self, gameState, myPos, minDist):

        explored, tempCost, Q = [], [], util.PriorityQueue()
        explored.append(myPos)
        tempCost.append(0)
        Q.push([myPos, []], minDist)
        while not Q.isEmpty():
            node, path = Q.pop()
            pathCost = tempCost[explored.index(node)]
            if self.isHomeSide(node, gameState):
                return path
            for successor in self.legalAction[node]:
                state, action, cost = successor
                if state not in explored:
                    childDist = float('inf')
                    explored.append(state)
                    tempCost.append(childDist)
                else:
                    childDist = tempCost[explored.index(state)]
                if pathCost + cost < childDist:
                    childDist = pathCost + cost
                    Q.update([state, path + [action]], childDist + minDist)
                    tempCost[explored.index(state)] = childDist
        return []

    def computeActionFromQValues(self, state):
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            value = self.getQValue(state, action)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP
        return random.choice(bestActions)

    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        bestValue = -999999
        noLegalActions = True
        for action in state.getLegalActions(self.index):
            noLegalActions = False
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        if noLegalActions:
            return 0
        return bestValue

    def getQValue(self, state, action):
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        return features * weights

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()
        return -len(foodList)

    def observationFunction(self, gameState):
        if len(self.observationHistory) > 0 and self.isTraining():
            self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
        return gameState.makeObservation(self.index)

    def isTraining(self):
        return self.episodesCur < self.numTraining

    def update(self, state, action, nextState, reward):

        difference = (reward + self.discount * self.computeValueFromQValues(nextState))
        difference -= self.getQValue(state, action)

        newWeights = self.weights.copy()

        features = self.getFeatures(state, action)
        for feature in features:

            newWeight = newWeights[feature] + self.alpha * difference * features[feature]
            newWeights[feature] = newWeight
        self.weights = newWeights.copy()

    def final(self, state):
        CaptureAgent.final(self, state)
        self.episodesCur += 1

class ReflexCaptureAgent(ApproximateQAgent):

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.yAxis = 0.0
        self.defenseTime = 0.0
        self.lastNumReturnedPellets = 0.0
        self.getLegalPositions(gameState)
        self.originalFood = len(self.getFood(gameState).asList())

    def initialBeliefs(self,gameState):
        for enemy in self.getOpponents(gameState):
            beliefs[enemy] = util.Counter()
            beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0
            beliefs[enemy].normalize()
        beliefsInitialized.append("initial")

    def __init__(self, index):
        self.actionHistory=[]
        self.count=0
        ApproximateQAgent.__init__(self, index)
        self.weights = util.Counter()
        self.weights['successorScore'] = 300
        self.weights['distToFood'] = -20
        self.weights['distToGhost'] = 2
        self.weights['capsulesValue'] = 20
        self.weights['capsulesScore'] = 500
        self.distanceToTrackcapsulesValue = 8
        self.weights['chaseEnemyValue'] = -30
        self.weights['catchPacman'] = 500
        self.weights['corner'] = -15
        self.weights['stop'] = -1000
        self.weights['legalActions'] = 100
        self.legalActionMap = {}
        self.legalPositionsInitialized = False

    def getLegalPositions(self, gameState):
        if not self.legalPositionsInitialized:
            self.legalPositions = []
            walls = gameState.getWalls()
            for x in range(walls.width):
                for y in range(walls.height):
                    if not walls[x][y]:
                        self.legalPositions.append((x, y))
            self.legalPositionsInitialized = True
        return self.legalPositions

    def getLegalActions(self, gameState):
        currentPos = gameState.getAgentPosition(self.index)
        if currentPos not in self.legalActionMap:
            self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
        return self.legalActionMap[currentPos]

    def getMiniDist(self,gameState,myPos,opponentsList):
        enemyInfo = []
        for index in opponentsList:
            pos = self.getMostLikelyGhostPosition(gameState,index)
            dis = self.getMazeDistance(myPos, pos)
            enemyInfo.append((pos,dis))
        return enemyInfo

    def getWeights(self):
        return self.weights

    def getcapsulesValue(self, myPos, successor, nonScaredGhosts):
        powerCapsules = self.getCapsules(successor)
        minDistance = 0
        if len(powerCapsules) > 0 and len(nonScaredGhosts) == 0:
            distances = [self.getMazeDistance(myPos, pellet) for pellet in powerCapsules]
            minDistance = min(distances)
        return max(self.distanceToTrackcapsulesValue - minDistance, 0)

    def getMostLikelyGhostPosition(self,gameState, ghostAgentIndex):
        allPossible = util.Counter()
        canGetPosition = gameState.getAgentPosition(ghostAgentIndex)
        if canGetPosition != None:
            return canGetPosition
        else:
            return beliefs[ghostAgentIndex].argMax()

    def getLegalActionModifier(self, gameState, numLoops):
        legalActions = self.getLegalActions(gameState)
        numActions = len(legalActions)
        for legalAction in legalActions:
            if numLoops > 0:
                newState = self.getSuccessor(gameState, legalAction)
                numActions += self.getLegalActionModifier(newState, numLoops - 1)
        return numActions

    def refreshProbability(self,gameState,opponent):
        newBeliefs = util.Counter()
        for p in self.getLegalPositions(gameState):
            newProbability = util.Counter()
            probPos = {}
            x, y = p
            probPos[1] = (x + 1, y)
            probPos[2] = (x, y + 1)
            probPos[3] = (x - 1, y)
            probPos[4] = (x, y - 1)
            probPos[5] = (x, y)
            for newP in probPos.values():
                if newP in self.getLegalPositions(gameState):
                    newProbability[p] = 1.0
            newProbability.normalize()
            for newPos, prob in newProbability.items():
                newBeliefs[newPos] += prob * (beliefs[opponent][newPos] + 0.0001)
        newBeliefs.normalize()
        beliefs[opponent] = newBeliefs

    def observeAllOpponents(self, gameState):
        if len(beliefsInitialized)==0:
            self.initialBeliefs(gameState)
        else:
            for opponentIndex in self.getOpponents(gameState):
                self.refreshProbability(gameState,opponentIndex)
            pacmanPosition = gameState.getAgentPosition(self.index)
            for opponent, belief in beliefs.items():
                noisyDistance = gameState.getAgentDistances()[opponent]
                newBeliefs = util.Counter()
                for p in self.getLegalPositions(gameState):
                    trueDistance = util.manhattanDistance(p, pacmanPosition)
                    distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance)
                    if gameState.getAgentState(opponent).isPacman and p not in self.startEdge(gameState):
                        if trueDistance > 5:
                            if self.isHomeSide(p,gameState):
                                newBeliefs[p] += (beliefs[opponent][p]+ 0.0001) * distanceProb
                    else:
                        newBeliefs[p] = 0
                newBeliefs.normalize()
                beliefs[opponent] = newBeliefs

    def startEdge(self,gameState):
        startEdge=[]
        x,y=self.start
        for i in range(1,gameState.data.layout.height):
            startEdge.append((x,i))
        return startEdge

    def observationFunction(self, gameState):
            return ApproximateQAgent.observationFunction(self, gameState)

    def chooseAction(self, gameState):
        self.observationHistory.append(gameState)
        self.observeAllOpponents(gameState)
        legalActions = gameState.getLegalActions(self.index)
        action = None
        ghosts,enemyPacmen,enemyIndexes = [],[],[]
        for i in self.getOpponents(gameState):
            if not gameState.getAgentState(i).isPacman:
                enemyIndexes.append(i)
                ghosts.append(gameState.getAgentPosition(i))
            if gameState.getAgentState(i).isPacman:
                enemyPacmen.append(i)

        agentPos = gameState.getAgentPosition(self.index)

        enemyDis = 999999
        enemyPos = None
        enemyInx=-1
        for ghost in enemyIndexes:
            ghostPos = gameState.getAgentPosition(ghost)
            if ghostPos != None:
                dis = self.getMazeDistance(agentPos, ghostPos)
                if dis < enemyDis:
                    enemyDis = dis
                    enemyPos = ghostPos
                    enemyInx = ghost

        foodlist = self.getFood(gameState).asList()

        if len(legalActions):
            action = self.computeActionFromQValues(gameState)

        carryFood = self.originalFood - len(foodlist)
        if self.isHomeSide(gameState.getAgentPosition(self.index), gameState):
            self.originalFood = self.originalFood - carryFood
            carryFood = 0
            self.originalFood = len(self.getFood(gameState).asList())

        self.lastAction = action
        if (len(ghosts) > 0 and not self.isHomeSide(agentPos, gameState)):
            if enemyDis <= 4 and gameState.getAgentState(enemyInx).scaredTimer <= 4:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 2)
                heruisticDist = self.breadthFirstSearch(agentPos, gameState)
                if heruisticDist > 0:
                    actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                    action = actionlist[0]
                    return action

        if carryFood > 9 and enemyInx != -1 and gameState.getAgentState(enemyInx).scaredTimer <= 4:
            if enemyPos != None:
                 self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
            heruisticDist = self.breadthFirstSearch(agentPos, gameState)
            if heruisticDist > 0:
                 actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                 action = actionlist[0]
                 return action
        if gameState.data.timeleft < 200 and carryFood >0:
            if enemyPos != None:
                 self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
            heruisticDist = self.breadthFirstSearch(agentPos, gameState)
            if heruisticDist > 0:
                 actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                 action = actionlist[0]
                 return action

        if len(foodlist) <= 2:
            if enemyPos!=None:
                self.refreshLegalActionAndCorners(gameState, enemyPos, 1)
            heruisticDist = self.breadthFirstSearch(agentPos, gameState)
            if heruisticDist > 0:
                actionlist = self.aStarSearch(gameState, agentPos, heruisticDist)
                action = actionlist[0]
                return action

        return action

    def breadthFirstSearch(self, myPos, gameState):
        explored, frontier = [], util.Queue()
        explored.append(myPos)
        frontier.push([myPos, []])
        while not frontier.isEmpty():
            node, path = frontier.pop()
            if self.isHomeSide(node, gameState):
                return len(path)
            for successor in self.legalAction[node]:
                nextState, action, cost = successor
                if nextState not in explored:
                    explored.append(nextState)
                    frontier.push([nextState, path + [action]])
        return 0

    def refreshLegalActionAndCorners(self, gameState, enemyPos, n):
        legalAction = {}
        corners = []
        walls = self.enemyDomain(gameState, enemyPos, n)
        for x in range(0, gameState.data.layout.width):
            for y in range(0, gameState.data.layout.height):
                if not gameState.hasWall(x, y):
                    possible = []
                    adjacentToConers = 0
                    for action, position in Actions._directionsAsList:
                        px, py = position
                        ny = py + y
                        nx = px + x
                        if not walls[nx][ny]: possible.append(((nx, ny), action, 1))
                    legalAction[(x, y)] = possible
                    if len(possible) <= 2:
                        corners.append((x, y))
        self.legalAction = legalAction
        self.corners = corners

    def enemyDomain(self, gameState, enemyPos, n):
        x, y = enemyPos
        x = int(x)
        y = int(y)
        walls = copy.deepcopy(gameState.getWalls())
        for i in range(0, n):
            walls[x + i][y] = True
            walls[x][y + i] = True
            walls[x + i][y + i] = True
            walls[x + i][y - i] = True
            walls[x - i][y] = True
            walls[x][y - i] = True
            walls[x - i][y - i] = True
            walls[x - i][y + i] = True
        return walls

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentPosition(self.index)

        ghosts, nonScaredGhosts, scaredGhosts,enemyPacmen = [],[],[] ,[]
        for i in self.getOpponents(gameState):
            if not successor.getAgentState(i).isPacman:
                if successor.getAgentPosition(i)!=None:
                    ghosts.append(successor.getAgentPosition(i))
                if successor.getAgentState(i).scaredTimer > 0:
                    nonScaredGhosts.append(i)
                else:
                    scaredGhosts.append(i)
            if successor.getAgentState(i).isPacman:
                enemyPacmen.append(i)
        enemyInfo = []

        if len(enemyPacmen)>0:
            features['catchPacman'] = -len(enemyPacmen)
            minPacmanPos[self.index] = self.getMiniDist(successor, myPos, enemyPacmen)

            panicDis=0
            if len(enemyPacmen)==1:
                if minPacmanPos[self.index] != None and minPacmanPos[self.teamIndex]== None:
                    panicDis = minPacmanPos[self.index][0][1]
                elif minPacmanPos[self.index]!= None and minPacmanPos[self.teamIndex]!= None:
                    if minPacmanPos[self.index][0][1] < minPacmanPos[self.teamIndex][0][1]:
                        panicDis = minPacmanPos[self.index][0][1]
                    elif minPacmanPos[self.index][0][1] == minPacmanPos[self.teamIndex][0][1]:
                        if self.index<=1:
                            panicDis = minPacmanPos[self.index][0][1]
            if len(enemyPacmen)==2:
                if minPacmanPos[self.index]!=None and minPacmanPos[self.teamIndex]!=None:
                    if  len(minPacmanPos[self.index])>1 and len(minPacmanPos[self.teamIndex])==1:
                        if util.manhattanDistance(minPacmanPos[self.index][1][0],minPacmanPos[self.teamIndex][0][0])<=3:
                            panicDis = minPacmanPos[self.index][1][1]
                        else:
                            panicDis = minPacmanPos[self.index][0][1]
                    elif  len(minPacmanPos[self.index])==1 and len(minPacmanPos[self.teamIndex])>1:
                        panicDis = minPacmanPos[self.index][0][1]
                    elif  len(minPacmanPos[self.index])>1 and len(minPacmanPos[self.teamIndex])>1:
                        if util.manhattanDistance(minPacmanPos[self.index][0][0],minPacmanPos[self.teamIndex][0][0])<=3:
                            if minPacmanPos[self.index][0][1] < minPacmanPos[self.teamIndex][0][1]:
                                panicDis = minPacmanPos[self.index][0][1]
                            else:
                                panicDis = minPacmanPos[self.index][1][1]
                        elif util.manhattanDistance(minPacmanPos[self.index][0][0], minPacmanPos[self.teamIndex][0][0])>3:
                            if minPacmanPos[self.index][0][1] < minPacmanPos[self.index][1][1]:
                                panicDis = minPacmanPos[self.index][0][1]
                            else:
                                panicDis = minPacmanPos[self.index][1][1]
                elif minPacmanPos[self.index] != None and minPacmanPos[self.teamIndex] == None:
                    if len(minPacmanPos[self.index])>1:
                        if minPacmanPos[self.index][0][1] < minPacmanPos[self.index][1][1]:
                            panicDis = minPacmanPos[self.index][0][1]
                        else:
                            panicDis = minPacmanPos[self.index][1][1]
                    else:
                        panicDis = minPacmanPos[self.index][0][1]

            features['chaseEnemyValue'] = panicDis

            if len(self.getFoodYouAreDefending(gameState).asList()) <= 10:
                features['chaseEnemyValue'] *= 100
        else:
            minPacmanPos[self.index]=None
            minPacmanPos[self.teamIndex] = None


        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        capsulesList = self.getCapsules(successor)
        features['capsulesScore'] = -len(capsulesList)

        if len(foodList) > 0:
            Distance = min([self.getMazeDistance(myPos, food) + abs(self.yAxis - food[1]) for food in foodList])
            features['distToFood'] = Distance

        defendCapsulesDist=self.getCapsulesYouAreDefending(successor)
        if len(defendCapsulesDist)>0:
            if len(enemyPacmen) >0:
                features['defendCapsulesDist'] = min([self.getMazeDistance(i,myPos) for i in defendCapsulesDist])

        features['capsulesValue'] = self.getcapsulesValue(myPos,successor, nonScaredGhosts)

        if len(scaredGhosts)>0 and len(ghosts)>0 and not self.isHomeSide(myPos,gameState):
            features['distToGhost'] = min([self.getMazeDistance(myPos,i) for i in ghosts])


        if myPos in self.corners:
            features['corner'] = 1

        if action == Directions.STOP:
            features['stop'] = 1
        features['legalActions'] = self.getLegalActionModifier(gameState, 1)

        if features['chaseEnemyValue']!=0:
            features['defendCapsulesDist']=0
            features['distToFood']=0
            features['capsulesValue']=0
        return features

class TopAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.yAxis = gameState.data.layout.height
        self.defenseTime = 0.0
        self.originalFood = len(self.getFood(gameState).asList())
        self.legalAction, self.corners = self.initialLegalAction(gameState)
        self.lastNumReturnedPellets = 0.0
        self.teamIndex = -1
        self.start = gameState.getAgentPosition(self.index)
        for i in self.getTeam(gameState):
            if i != self.index:
                self.teamIndex = i
                minPacmanPos[i] = None
                minGhostPos[i]=None

class BottomAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.yAxis = 0.0
        self.start = gameState.getAgentPosition(self.index)
        self.originalFood = len(self.getFood(gameState).asList())
        self.defenseTime = 0.0
        self.lastNumReturnedPellets = 0.0
        self.legalAction, self.corners = self.initialLegalAction(gameState)
        self.start = gameState.getAgentPosition(self.index)
        for i in self.getTeam(gameState):
            if i != self.index:
                self.teamIndex = i
                minPacmanPos[i] = None
                minGhostPos[i] = None

