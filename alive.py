# myTeam.py
# ---------
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


from captureAgents import CaptureAgent, AgentFactory
import random, time, util
from game import Directions, Agent
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='TopAgent', second = 'BAgent', numTraining=0):
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

POWERCAPSULETIME = 120


class MCTSAgent(CaptureAgent):
  def __init__(self, index, epsilon=0.01, timeForComputing=.1, depth=3, times=20, alpha=0.9, center = (0, 0), atCenter = False):
    self.depth = depth
    self.times = times
    self.alpha = float(alpha)
    self.powerTimer = 0
    self.center = center
    self.atCenter = atCenter

    self.epilon = float(epsilon)
    CaptureAgent.__init__(self, index, timeForComputing=.1)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.goToCenter(gameState)

  def runbackCheck(self, state):
    myState = state.getAgentState(self.index)
    myPos = myState.getPosition()
    enemy_1 = state.getAgentState(self.getOpponents(state)[0])
    enemy_2 = state.getAgentState(self.getOpponents(state)[1])
    dis = 1000
    if enemy_1.getPosition() != None:
      dis = self.getMazeDistance(myPos, enemy_1.getPosition())
    if enemy_2.getPosition() != None:
      dis = min(dis, self.getMazeDistance(myPos, enemy_2.getPosition()))

    if (dis < 3 and state.getAgentState(self.index).isPacman):
      return True
    else:
      return False

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def chooseAction(self, gameState):

    actions = gameState.getLegalActions(self.index)
    goodactions = []
    for a in actions:
      if a != 'Stop':
        goodactions.append(a)

    if util.flipCoin(self.epilon):
      return random.choice(goodactions)
    else:
      return self.computeAction(gameState)

  def computeAction(self, state):
    actions = state.getLegalActions(self.index)
    goodactions = []
    for x in actions:
      if x != 'Stop':
        goodactions.append(x)
    values = [self.mainevaluate(state, a) for a in goodactions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(goodactions, values) if v == maxValue]

    return random.choice(bestActions)

  def mainevaluate(self, state, action):
    nextState = state.generateSuccessor(self.index, action)
    nextStateValue = self.getnextStepBasicValue(state, action)

    lefttimes = self.times
    while (lefttimes > 0):
      nextStateValue += self.deepsearch(nextState, self.depth)
      lefttimes = lefttimes - 1
    return nextStateValue

  def deepsearch(self, state, depth):
    onetimevalue = 0
    newstate = state
    while (depth > 0):
      fakeactions = newstate.getLegalActions(self.index)
      fakeaction = random.choice(fakeactions)
      onetimevalue += self.getbasicValue(newstate, fakeaction) * self.alpha
      newstate = newstate.generateSuccessor(self.index, fakeaction)
      depth = depth - 1
    return onetimevalue

  def getnextStepBasicValue(self, state, action):
    features = self.getNextfeatures(state, action)
    weights = self.getNextweights(state, action)
    value = 0

    for feature in features:
      value += features[feature] * weights[feature]

    return value

  def getNextfeatures(self, state, action):
    features = util.Counter()
    nextState = state.generateSuccessor(self.index, action)
    actions = nextState.getLegalActions(self.index)

    myState = nextState.getAgentState(self.index)
    myPos = myState.getPosition()
    myteam = self.getTeam(state)
    teammate_index = 0
    for index in myteam:
      if index != self.index:
        teammate_index = index
    matePos = state.getAgentState(teammate_index).getPosition()

    foods = self.getFood(nextState).asList()
    ourfoods = self.getFoodYouAreDefending(nextState).asList()

    features['eatfood'] = len(foods)
    features['gost-one-step'] = 0
    features['eatPacman'] = 0
    features['runback'] = 0

    Capsules = self.getCapsules(state)

    # set Capsules
    if myPos in Capsules:
      self.powerTimer = POWERCAPSULETIME

    if self.powerTimer > 0:
      self.powerTimer -= 1

    enemyIndexList = self.getOpponents(state)
    for enemyIndex in enemyIndexList:
      enemy = state.getAgentState(enemyIndex)
      if enemy.getPosition() != None:
        enemyActions = state.getLegalActions(enemyIndex)
        enemyPosiblePos = []
        for enemyAction in enemyActions:
          enemyNextState = state.generateSuccessor(enemyIndex, enemyAction)
          pos = enemyNextState.getAgentState(enemyIndex).getPosition()
          enemyPosiblePos.append(pos)
        Mydis = self.getMazeDistance(myPos, enemy.getPosition())
        Matedis = self.getMazeDistance(matePos, enemy.getPosition())
        if enemy.isPacman:
          if Mydis < Matedis:
            features['eatPacman'] = self.getMazeDistance(myPos, enemy.getPosition())
        elif (myPos in enemyPosiblePos):

          minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
          features['runback'] = minDistance
          features['distanceToFood'] = 0
          features['gost-one-step'] = -100

    # for enemy in [enemy_1, enemy_2]:
    #   if enemy.getPosition()!=None:
    #     # print ('mypos', myPos)
    #     # print ('enemy', enemy.getPosition())
    #     Mydis = self.getMazeDistance(myPos,enemy.getPosition())
    #     Matedis = self.getMazeDistance(matePos,enemy.getPosition())
    #     if enemy.isPacman:
    #       if Mydis<Matedis:
    #         features['eatPacman'] = self.getMazeDistance(myPos,enemy.getPosition())
    #     elif(myPos == enemy.getPosition()):
    #       print('run!')
    #       minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
    #       features['runback'] = minDistance
    #       features['distanceToFood'] = 0
    #       features['gost-one-step'] = -100

    if len(foods) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foods])
      features['distanceToFood'] = minDistance

    # if (nextState.getAgentState(self.index).numCarrying > 0 and self.runbackCheck(nextState) or
    #             nextState.getAgentState(self.index).numCarrying > 3):
    #     features['distanceToFood'] = 0
    #     features['eatfood'] = 0
    #     backdistance = 0
    #     if nextState.getAgentState(self.index).numCarrying > 0 and self.runbackCheck(nextState):
    #         enemy_1 = state.getAgentState(self.getOpponents(state)[0])
    #         enemy_2 = state.getAgentState(self.getOpponents(state)[1])
    #         position = (0, 0)
    #         if enemy_1.getPosition() != None:
    #             position = enemy_1.getPosition()
    #         else:
    #             position = enemy_2.getPosition()
    #
    #         was = nextState.getWalls().asList(False)
    #         walls = was + [position]
    #         backdistance = min([self.nodeaddistance(myPos, ourfood, walls) for ourfood in ourfoods])
    #         if features['gost-one-step']==-100:
    #           print backdistance
    #     else:
    #         backdistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
    #     features['runback'] = backdistance

    if (
          (nextState.getAgentState(self.index).numCarrying > 0 and self.runbackCheck(
            nextState)) or nextState.getAgentState(
          self.index).numCarrying > 3):
      features['eatfood'] = 0
      if len(Capsules) > 0:
        backdistance = min([self.getMazeDistance(myPos, Capsule) for Capsule in Capsules])
        features['runback'] = backdistance * 10000
      else:
        backdistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
        features['runback'] = backdistance
      features['distanceToFood'] = 0
      features['eatfood'] = 0

    features['notdeadend'] = 0
    if (self.runbackCheck(nextState) and len(actions) == 1):
      features['notdeadend'] = 1

    # features['toCenter'] = 0
    # features['distanceToCenter'] = 0
    # if self.atCenter == False:
    #   dist = self.getMazeDistance(myPos, self.center)
    #   features['distanceToCenter'] = dist
    #   features['toCenter'] = 1
    # if myPos == self.center and self.atCenter == False:
    #   self.atCenter = True

    # Compute score from successor state
    features['successorScore'] = self.getScore(nextState)
    if myPos in self.getCapsules(state):
      self.powerTimer = POWERCAPSULETIME

    # If powered, reduce power timer each itteration
    if self.powerTimer > 0:
      self.powerTimer -= 1

    if (self.isPowered()):
      features['isPowered'] = self.powerTimer / POWERCAPSULETIME
      features['eatfood'] = 10 * len(foods)
      features['distanceToFood'] = 10 * min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
      features['gost-one-step'] = 0
    else:
      features['isPowered'] = 0.0

    return features

  def getNextweights(self, state, action):
    return {'successorScore': 200, 'eatPacman': -10, 'gost-one-step': 1000, 'distanceToFood': -1, 'eatfood': -10,
            'runback': -0.05, 'notdeadend': 0, 'isPowered': 5000000}

  def isPowered(self):
    return self.powerTimer > 0

  def getbasicValue(self, state, action):
    features = self.getfeatures(state, action)
    weights = self.getweights(state, action)
    value = 0

    for feature in features:
      value += features[feature] * weights[feature]

    return value

  def getfeatures(self, state, action):
    features = util.Counter()
    nextState = state.generateSuccessor(self.index, action)
    actions = state.getLegalActions(self.index)

    myState = nextState.getAgentState(self.index)
    myPos = myState.getPosition()
    myteam = self.getTeam(state)
    teammate_index = 0
    for index in myteam:
      if index != self.index:
        teammate_index = index
    matePos = state.getAgentState(teammate_index).getPosition()
    # walls = state.getWalls().asList()
    # print walls



    enemy_1 = nextState.getAgentState(self.getOpponents(nextState)[0])
    enemy_2 = nextState.getAgentState(self.getOpponents(nextState)[1])
    foods = self.getFood(nextState).asList()
    ourfoods = self.getFoodYouAreDefending(nextState).asList()

    features['eatfood'] = len(foods)
    features['gost-one-step'] = 0
    features['eatPacman'] = 0
    features['runback'] = 0
    if enemy_1.getPosition() != None:
      Mydis = self.getMazeDistance(myPos, enemy_1.getPosition())
      Matedis = self.getMazeDistance(matePos, enemy_1.getPosition())
      if enemy_1.isPacman and Mydis < Matedis:
        features['eatPacman'] = self.getMazeDistance(myPos, enemy_1.getPosition())
      elif ((not enemy_1.isPacman) and myPos == enemy_1.getPosition()):
        print "run"
        minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
        features['runback'] = minDistance
        features['distanceToFood'] = 0
        features['gost-one-step'] = -1

    if enemy_2.getPosition() != None:
      Mydis = self.getMazeDistance(myPos, enemy_2.getPosition())
      Matedis = self.getMazeDistance(matePos, enemy_2.getPosition())
      if enemy_2.isPacman and Mydis < Matedis:
        features['eatPacman'] = self.getMazeDistance(myPos, enemy_2.getPosition())
      elif ((not enemy_1.isPacman) and myPos == enemy_2.getPosition()):
        print "run"
        minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
        features['runback'] = minDistance
        features['distanceToFood'] = 0
        features['gost-one-step'] = -1

    if len(foods) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foods])
      features['distanceToFood'] = minDistance

    if (nextState.getAgentState(self.index).numCarrying > 2 and self.runbackCheck(nextState) or
            nextState.getAgentState(self.index).numCarrying > 3):
      features['eatfood'] = 0

      backdistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
      features['runback'] = backdistance
      features['distanceToFood'] = 0
      features['eatfood'] = 0

    features['notdeadend'] = 0
    if (self.runbackCheck(nextState) and len(actions) == 2):
      features['notdeadend'] = 1

    features['toCenter'] = 0
    features['distanceToCenter'] = 0
    if self.atCenter == False:
      dist = self.getMazeDistance(myPos, self.center)
      features['distanceToCenter'] = dist
      features['toCenter'] = 1
    if myPos == self.center and self.atCenter == False:
      self.atCenter = True

    # Compute score from successor state
    features['successorScore'] = self.getScore(nextState)
    return features

  def getweights(self, state, action):
    return {'successorScore': 200, 'eatPacman': -10, 'gost-one-step': 10, 'distanceToFood': -1, 'eatfood': -10,
            'runback': -0.05, 'notdeadend': 0, 'distanceToCenter': -1, 'toCenter': 1000}


class TopAgent(MCTSAgent):
  def goToCenter(self, gameState):
    locations = []
    self.atCenter = False
    x = gameState.getWalls().width / 2
    y = gameState.getWalls().height / 2

    if self.red:
      x = x - 1

    self.center = (x, y)
    print self.center
    maxHeight = gameState.getWalls().height

    for i in xrange(maxHeight - y):
      if not gameState.hasWall(x, y):
        locations.append((x, y))
      y = y + 1

    myPos = gameState.getAgentState(self.index).getPosition()
    minDist = float('inf')
    minPos = None


    for location in locations:
      dist = self.getMazeDistance(myPos, location)
      if dist <= minDist:
        minDist = dist
        minPos = location

    self.center = minPos

class BottomAgent(MCTSAgent):
  def goToCenter(self, gameState):
    locations = []
    self.atCenter = False
    x = gameState.getWalls().width / 2
    y = gameState.getWalls().height / 2
    # 0 to x-1 and x to width
    if self.red:
      x = x - 1
    # Set where the centre is
    self.center = (x, y)

    # Look for locations to move to that are not walls (favor bot positions)
    for i in xrange(y):
      if not gameState.hasWall(x, y):
        locations.append((x, y))
      y = y - 1

    myPos = gameState.getAgentState(self.index).getPosition()
    minDist = float('inf')
    minPos = None

    # Find shortest distance to centre
    for location in locations:
      dist = self.getMazeDistance(myPos, location)
      if dist <= minDist:
        minDist = dist
        minPos = location

    self.center = minPos

class BAgent(BottomAgent):
  def getfeatures(self, state, action):
    features = util.Counter()
    nextState = state.generateSuccessor(self.index, action)
    actions = state.getLegalActions(self.index)
    myPos = nextState.getAgentState(self.index).getPosition()
    enemy_1 = nextState.getAgentState(self.getOpponents(nextState)[0])
    enemy_2 = nextState.getAgentState(self.getOpponents(nextState)[1])
    foods = self.getFood(nextState).asList()
    ourfoods = self.getFoodYouAreDefending(nextState).asList()
    bottomfoods = self.leasty(foods)
    ourbottomfoods = self.leasty(ourfoods)

    myteam = self.getTeam(state)
    teammate_index = 0
    for index in myteam:
      if index != self.index:
        teammate_index = index
    matePos = state.getAgentState(teammate_index).getPosition()

    features['eatfood'] = len(foods)

    features['runback'] = 0
    features['gost-one-step'] = 0
    features['eatPacman'] = 0
    if enemy_1.getPosition() != None:
      Mydis = self.getMazeDistance(myPos, enemy_1.getPosition())
      Matedis = self.getMazeDistance(matePos, enemy_1.getPosition())
      if enemy_1.isPacman and Mydis < Matedis:
        features['eatPacman'] = self.getMazeDistance(myPos, enemy_1.getPosition())
      elif (myPos == enemy_1.getPosition()):
        print "run"
        minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
        features['runback'] = minDistance
        features['distanceToFood'] = 0
        features['gost-one-step'] = -1

    if enemy_2.getPosition() != None:
      Mydis = self.getMazeDistance(myPos, enemy_2.getPosition())
      Matedis = self.getMazeDistance(matePos, enemy_2.getPosition())
      if enemy_2.isPacman and Mydis < Matedis:
        features['eatPacman'] = self.getMazeDistance(myPos, enemy_2.getPosition())
      elif (myPos == enemy_2.getPosition()):
        print "run"
        backdistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourfoods])
        features['runback'] = backdistance
        features['distanceToFood'] = 0
        features['gost-one-step'] = -1

    if len(foods) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in bottomfoods])
      features['distanceToFood'] = minDistance

    if ((nextState.getAgentState(self.index).numCarrying > 1 and self.runbackCheck(nextState)) or
            nextState.getAgentState(self.index).numCarrying > 3):
      features['eatfood'] = 0
      minDistance = min([self.getMazeDistance(myPos, ourfood) for ourfood in ourbottomfoods])
      features['runback'] = minDistance
      features['distanceToFood'] = 0
      features['eatfood'] = 0

    features['notdeadend'] = 0
    if (self.runbackCheck(nextState) and len(actions) == 2):
      features['notdeadend'] = 1

    # Compute score from successor state
    features['successorScore'] = self.getScore(nextState)
    return features

  def getweights(self, state, action):
    return {'successorScore': 200, 'eatPacman': -10, 'gost-one-step': 100, 'distanceToFood': -1, 'eatfood': -10,
            'runback': -0.05, 'notdeadend': 0}

  def leasty(self, foodlist):
    if foodlist != None:

      minimal = foodlist[0][1]
      lindex = []

      for i in range(len(foodlist)):
        if foodlist[i][1] < minimal:
          minimal = foodlist[i][1]
          lindex[:] = []
          lindex.append(foodlist[i])
        elif foodlist[i][1] == minimal:
          lindex.append(foodlist[i])

      return lindex
