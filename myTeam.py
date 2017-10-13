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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from util import pause
from capture import noisyDistance
import random
import math
import hashlib
import logging
import argparse

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'DefenseAgent'):
               # first = 'AttackAgent', second = 'DefenseAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class MCTSAgent(CaptureAgent):
  """
  A Monte-Carlo Tree Search base Agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()

    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # Once attacker capture food, then it go back to home
    agentState = gameState.getAgentState(self.index)

    if agentState.numCarrying > 1:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # Once total food left <= 2, then they go back to home
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class AttackAgent(MCTSAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    # pause()
    # Compute distance to the nearest ghost
    # x = self.getOpponents(gameState)
    x = self.getOpponents(successor)
    enemies = [successor.getAgentState(i) for i in x]
    # enemies = [gameState.getAgentState(i) for i in x]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
    # print ghosts

    # ghosts = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    # features['noisyDistanceToGhost']
    # enemiesPosition = [a.getPosition() for a in enemies]
    # dist1, dist2 = noisyDistance(myPos, enemies.getPosition())

    if len(ghosts) > 0:
      myPos = successor.getAgentPosition(self.index)
      # myPos = gameState.getAgentPosition(self.index)
      positions = [agent.getPosition() for agent in ghosts]
      self.target = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      self.distanceToGhost = self.getMazeDistance(myPos, self.target)  
      if self.distanceToGhost < 10:
        features['distanceToGhost'] = self.distanceToGhost
      print 'Ghost Positions:', positions  # Ghost index: 1, 3
      print 'Pacman Position:', myPos     # Pacman index 0
      print 'distanceToGhost', features['distanceToGhost']     
    else:
      # print 'noisyDistanceToGhost1', dist1, 'noisyDistanceToGhost1', dist2
      features['distanceToGhost'] = 0
      print 'no ghost'

    # Compute distance to the nearest Power Pills
    # feature['distanceToPower']
    

    # print self.getPreviousObservation()

    return features

  def getWeights(self, gameState, action):

    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 10}

class DefenseAgent(MCTSAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

########  Monte Carlo Simulation
  # def randomSimulation(self, depth, gameState):
  #   """
  #   Random simulate some actions for the agent. The actions other agents can take
  #   are ignored, or, in other words, we consider their actions is always STOP.
  #   The final state from the simulation is evaluated.
  #   """
  #   new_state = gameState.deepCopy()
  #   while depth > 0:
  #     # Get valid actions
  #     actions = new_state.getLegalActions(self.index)
  #     # The agent should not stay put in the simulation
  #     actions.remove(Directions.STOP)
  #     current_direction = new_state.getAgentState(self.index).configuration.direction
  #     # The agent should not use the reverse direction during simulation
  #     reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
  #     if reversed_direction in actions and len(actions) > 1:
  #       actions.remove(reversed_direction)
  #     # Randomly chooses a valid action
  #     a = random.choice(actions)
  #     # Compute new state and update depth
  #     new_state = new_state.generateSuccessor(self.index, a)
  #     depth -= 1
  #   # Evaluate the final simulation state
  #   return self.evaluate(new_state, Directions.STOP)

  # def takeToEmptyAlley(self, gameState, action, depth):
  #   """
  #   Verify if an action takes the agent to an alley with
  #   no pacdots.
  #   """
  #   if depth == 0:
  #     return False
  #   old_score = self.getScore(gameState)
  #   new_state = gameState.generateSuccessor(self.index, action)
  #   new_score = self.getScore(new_state)
  #   if old_score < new_score:
  #     return False
  #   actions   = new_state.getLegalActions(self.index)
  #   actions.remove(Directions.STOP)
  #   reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
  #   if reversed_direction in actions:
  #     actions.remove(reversed_direction)
  #   if len(actions) == 0:
  #     return True
  #   for a in actions:
  #     if not self.takeToEmptyAlley(new_state, a, depth - 1):
  #       return False
  #   return True

  # def chooseAction(self, gameState):
  #   """
  #   Picks among the actions with the highest Q(s,a) from history
  #   """
  #   # You can profile your evaluation time by uncommenting these lines
  #   #start = time.time()

  #   # Get valid actions. Staying put is almost never a good choice, so
  #   # the agent will ignore this action.
  #   all_actions = gameState.getLegalActions(self.index)
  #   all_actions.remove(Directions.STOP)
  #   actions = []
  #   for a in all_actions:
  #     if not self.takeToEmptyAlley(gameState, a, 5):
  #       actions.append(a)
  #   if len(actions) == 0:
  #     actions = all_actions

  #   fvalues = []
  #   for a in actions:
  #     new_state = gameState.generateSuccessor(self.index, a)
  #     value = 0
  #     for i in range(1,31):
  #       value += self.randomSimulation(10, new_state)
  #     fvalues.append(value)

  #   best = max(fvalues)
  #   ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
  #   toPlay = random.choice(ties)[1]

  #   #print 'eval time for offensive agent %d: %.4f' % (self.index, time.time() - start)
  #   return toPlay

########  End of Monte Carlo Simulation

  # def __init__(self):
  #   super(DefenseAgent, self).__init__()
  #   self.start = None

#######  Monte Carlo Tree Search Simulation
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

    # self.start = gameState.getAgentPosition(self.index)    
    # self.start = gameState.getAgentPosition(self.index)
    
    self.levels=3
    self.numSims=5
    self.svalue=0 
    self.smoves=[]
    self.sturns=5
    self.gameState = gameState
    self.current_node = Node( State(self.gameState, self.index, self.svalue, self.smoves, self.sturns) )
    print 'self.current_node', self.current_node
    # print 'Start location', self.current_node.state.gameState.getAgentState(self.index).getPosition()
    self.startPosition = self.current_node.state.gameState.getAgentState(self.index).getPosition()
    print 'Start location', self.startPosition

  def chooseAction(self, gameState):

    return self.runSimulation(self.current_node, gameState, self.index, self.levels, self.numSims)

    # actions = gameState.getLegalActions(self.index)
    # # You can profile your evaluation time by uncommenting these lines
    # # start = time.time()
    # values = [self.evaluate(gameState, a) for a in actions]
    # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue]

  def runSimulation(self, current_node, gameState, index, levels=3, numSims=5):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # pause()
    print 'runSim/////////////////////////////////////////////'
    self.gameState = gameState
    self.index = index
    current_node.state.setState(self.gameState, self.index)
    self.current_node = current_node
    print 'Current Location', self.current_node.state.gameState.getAgentState(self.index).getPosition()
    ### This location move!!
    self.index = index

    # for l in range(levels):
    #   print '*****level', l    
    #   current_node=UCTSEARCH(numSims/(l+1), self.current_node, self.index)
    #   # print("level %d"%l)
    #   print 'Current_node', current_node
    #   print 'Current_node.children', current_node.children
    #   if len(current_node.children) > 0:
    #     print 'Current_node.children Location', current_node.children[0].state.gameState.getAgentState(self.index).getPosition()
    #   print("Num Children: %d"%len(current_node.children))
    #   reward = 0
    #   for i,c in enumerate(current_node.children):
    #     print 'i: ', i # i = children num
    #     print 'c: ', c # c = node info, 2100
    #     print 'c.state.reward', c.state.reward() # 1900, reward is different from c
    #     if c.state.reward() > reward: 
    #       print 'c.state.reward in comparison', c.state.reward()
    #       reward = c.state.reward()
    #       self.current_node = c
    #     # print(i,c)
    #   print("Best Child: %s"%current_node.state)
    #   print("-------------------------------------------------------------") 

    # print '***Current_node.state', current_node.state
    # # self.current_node = current_node
    # print '***Current_node.state.move', current_node.state.move
    # # print '***Current_node.state.moveSeries', current_node.state.moveSeries
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # return current_node.state.move
    # # return current_node.state.moveSeries[0]

    self.current_node.children = []
    l = 0 
    child_node=UCTSEARCH(numSims/(l+1), self.current_node, self.index)
    print 'Child Location', child_node.state.gameState.getAgentState(self.index).getPosition()
    print 'Child_node.state.fromMove', child_node.state.fromMove

    return child_node.state.fromMove



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


########  End of Monte Carlo Tree Search Simulation


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    # Invader = Enemy in the vision
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [distanceCalculator.Distancer(gameState.data.layout).getDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    # In the begining, leave home as far as possible
    print 'myPos', myPos
    startPos = gameState.getInitialAgentPosition(self.index)
    # startPos = (1.0, 1.0)
    print 'startPos', startPos
    if len(invaders) > 0:
      features['distToHome'] = 0
    else:
      # features['distToHome'] = self.getMazeDistance(self.startPosition,myPos)
      # features['distToHome'] = self.getMazeDistance(startPos, myPos)
      features['distToHome'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(startPos, myPos)
      # features['distToHome'] = self.getMazeDistance(self.start, myPos)
   


    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToHome': 10}
    # return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToHome': 3}

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

# def runSimulation(self, gameState, levels=3, numSims=5):
#   """
#   Finds the next successor which is a grid position (location tuple).
#   """
#   current_node=Node( MCState( gameState ) )
#   for l in range(levels):
#     current_node=UCTSEARCH(numSims/(l+1),current_node)
#     print("level %d"%l)
#     print("Num Children: %d"%len(current_node.children))
#     for i,c in enumerate(current_node.children):
#       print(i,c)
#     print("Best Child: %s"%current_node.mcstate)
    
#     print("--------------------------------") 

class State():
  NUM_TURNS = 5
  GOAL = 0
  # MOVES=[2,-2,3,-3]
  # MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
  # num_moves=len(MOVES)
  # def __init__(self, gameState, index, value=0, moves=[], turn=NUM_TURNS):
  def __init__(self, gameState, index, value=0, move = None, fromMove = None, turn=NUM_TURNS):  
    self.gameState=gameState
    self.index=index
    # print 'index:', index  # index = 2
    self.value=value
    self.turn=turn
    # self.moves=moves
    # self.moves.append(move)
    # self.moveSeries=[]
    self.move=move
    self.fromMove=fromMove

  def setState(self, gameState, index):
    self.gameState=gameState
    self.index=index

  def next_state(self):
    # nextmove=random.choice([x*self.turn for x  in self.MOVES])
    # next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)
    # return next
    print '|||||||next_state'
    print '-------gameState', self.gameState.getAgentState(self.index).getPosition()
    print '-------value', self.value

    actions = self.gameState.getLegalActions(self.index)
    # The agent should not stay STOP
    actions.remove(Directions.STOP)

    print '+++++Legal actions', actions
    nextmove = random.choice([x for x in actions])
    self.move = nextmove
    # print 'nextmove', nextmove
    # self.moveSeries.append(nextmove)    # not used here
    # print 'self.moveSeries', self.moveSeries

    # nextGameState = DefenseAgent(self.index, 0.1).generateSuccessor(self.gameState, nextmove)
    nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
    # values = DefenseAgent(self.gameState).evaluate(self.gameState, nextmove)
    nextValue = DefenseAgent(self.index, 0.1).evaluate(self.gameState, nextmove)
    # print 'values', value
    # self.value = self.value + nextValue
    # print 'self.value', self.value

    nextState = State(nextGameState, self.index, nextValue, nextmove, nextmove, self.turn-1)
    # nextState = State(nextGameState, self.index, nextValue, self.moveSeries, self.turn-1)
    # nextState = State(self.gameState, self.index, self.value, self.moveSeries, self.turn-1)
    # self.turn = self.turn-1
    # print 'next', next
    print '-------self.move', self.move
    print '-------nextGameState', nextGameState.getAgentState(self.index).getPosition()
    # print '-------nextValue', nextValue
    print '-------nextState', nextState
    # print '-------self.moveSeries', self.moveSeries
    
    return nextState
    
  def terminal(self):
    if self.turn == 0:
      return True
    return False
  def reward(self):
    # r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
    r = self.value
    return r
  def __hash__(self):
    # return int(hashlib.md5(str(self.moves)).hexdigest(),16)
    return int(hashlib.md5(str(self.move)).hexdigest(),16)
  def __eq__(self,other):
    if hash(self)==hash(other):
      return True
    return False
  def __repr__(self):
    # s="Value: %d; Moves: %s"%(self.value,self.moves)
    s="Value: %d; Move: %s"%(self.value,self.move)
    return s
  


class Node():
  def __init__(self, state, parent=None):
    self.visits=1
    self.reward=0.0 
    self.state=state
    self.children=[]
    self.parent=parent  
  def add_child(self,child_state):
    child=Node(child_state,self)
    self.children.append(child)
  def update(self,reward):
    self.reward+=reward
    self.visits+=1
  def fully_expanded(self):
    # if len(self.children)==self.state.num_moves:
    actions = self.state.gameState.getLegalActions(self.state.index)
    actions.remove(Directions.STOP)
    availableActions = len(actions)
    print 'availableActions', availableActions
    print actions
    if len(self.children) == availableActions:  
      return True
    return False
  def __repr__(self):
    s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
    return s
    


def UCTSEARCH(budget,root,index):
  print 'UCTSEARCH'
  # print 'location', root.state.gameState.getAgentPosition(index)
  for iter in range(budget):
    if iter%10000==9999:
      logger.info("simulation: %d"%iter)
      logger.info(root)
    front=TREEPOLICY(root,index)
    reward=DEFAULTPOLICY(front.state)
    BACKUP(root,front,reward)
  # print 'root location', root.state.gameState.getAgentPosition(index)
  # print 'c location', BESTCHILD(root,0,index).state.gameState.getAgentPosition(index)
  return BESTCHILD(root,0,index)

def TREEPOLICY(node,index):
  print 'TREEPOLICY'
  while node.state.terminal()==False:
    # print 'terminal()==False'
    if node.fully_expanded()==False:  
      # print 'fully_expanded()==False'
      return EXPAND(node)
    else:
      node=BESTCHILD(node,SCALAR,index)
  return node

def EXPAND(node):
  print 'EXPAND'
  tried_children_state=[c.state for c in node.children]
  print 'tried_children_state', tried_children_state
  new_state=node.state.next_state()
  # print 'new_state', new_state
  counter = 0
  while new_state in tried_children_state:
    actions = node.state.gameState.getLegalActions(node.state.index)
    actions.remove(Directions.STOP)
    print 'new_state in tried, len(actions):', len(actions)
    # print 'node location', node.state.gameState.getAgentState(node.state.index).getPosition()
    if len(actions) == 1:   # if it only has one possible direction, then jump out
      break
    new_state=node.state.next_state()
    counter = counter+1   # still not solved
    if counter > 5:
      break
    # print 'new_state', new_state   
  print 'Until new_state not in tried_children_state', new_state
  node.add_child(new_state)
  for child in node.children:
    print 'node.children.state', child.state
  # print 'new_state in tried_children_state', new_state
  return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar,index):
  print 'BESTCHILD'
  print 'node location', node.state.gameState.getAgentPosition(index)
  bestscore=0.0
  bestchildren=[]
  for c in node.children:
    print 'c location', c.state.gameState.getAgentPosition(index)
    exploit=c.reward/c.visits
    explore=math.sqrt(math.log(2*node.visits)/float(c.visits))  
    score=exploit+scalar*explore
    if score==bestscore:
      bestchildren.append(c)  # more than one best child
    if score>bestscore:       # best child
      bestchildren=[c]
      bestscore=score
  if len(bestchildren)==0:
    logger.warn("OOPS: no best child found, probably fatal")

  # print 'index', index
  if bestchildren!=[] and index==2:
    # return random.choice(bestchildren)
    print 'bestchildren', bestchildren
    return bestchildren[0]
  else:
    return bestchildren

def DEFAULTPOLICY(state):
  print 'DEFAULTPOLICY'
  while state.terminal()==False:
    # print 'state.terminal()==False'
    state=state.next_state()
  return state.reward()

def BACKUP(root,node,reward):
  print 'BACKUP'
  while node!=None:
    node.visits+=1
    node.reward+=reward
    node=node.parent
    # node.parent = root
    # if node!=None:
      # print 'node.parent.state', node.state
  return


