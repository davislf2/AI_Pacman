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
              # first = 'DefenseAgent', second = 'DefenseAgent'):
               # first = 'AttackAgent', second = 'AttackAgent'):

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
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print '++++++++++++++++attacker eval time for agent %d: %.4f' % (self.index, time.time() - start)

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

    # Compute distance to the nearest ghost
    x = self.getOpponents(successor)
    enemies = [successor.getAgentState(i) for i in x]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

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

#######  Monte Carlo Tree Search Simulation
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

    self.levels=3
    self.numSims=10
    self.svalue=0 
    self.smoves=[]
    self.sturns=10
    self.gameState = gameState
    self.current_node = Node( MState(self.gameState, self.index, self.svalue, self.smoves, self.sturns) )
    print '~~~register self.current_node', self.current_node
    self.startPosition = self.current_node.mstate.gameState.getAgentState(self.index).getPosition()
    print '~~~register Start location', self.startPosition

  def chooseAction(self, gameState):

    return self.runSimulation(self.current_node, gameState, self.index, self.levels, self.numSims)

  def runSimulation(self, current_node, gameState, index, levels=3, numSims=5):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # pause()
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    
    print 'runSim/////////////////////////////////////////////'
    self.gameState = gameState
    self.index = index  
    self.current_node = current_node

    value = 0
    self.current_node.resetNode()
    self.current_node.mstate.resetMState(self.gameState, index, value)

    # print 'self.current_node', self.current_node, 'type', type(self.current_node)
    print 'Current Location', self.current_node.mstate.gameState.getAgentState(self.index).getPosition(), 'type', type(self.current_node.mstate.gameState.getAgentState(self.index).getPosition())
    ### This location move!!
    self.index = index

    self.current_node.children = []

    l = 0 
    child_node=UCTSEARCH(numSims/(l+1), self.current_node, self.index)
    print 'Child Location', child_node.mstate.gameState.getAgentState(self.index).getPosition()
    print 'Child_node.mstate.fromMove', child_node.mstate.fromMove

    print '++++++++++++++++defender eval time for agent %d: %.4f' % (self.index, time.time() - start)

    return child_node.mstate.fromMove


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

  def getManualWeights(self):
        
    return {'closest-food':-100, 'eats-food':100,'bias':3,'#-of-ghosts-1-step-away':-500,
            'distance-to-base':-100, 'distance-to-eat': -40, 'eat-pacman':200, 'avoid-pacman-power':-500,
            'distance-to-mate': 0.18, 'distance-to-capsule': -30, 'eat-capsule':200, 'eat-scared-ghost': 200}
  

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
    # print 'myPos', myPos
    startPos = gameState.getInitialAgentPosition(self.index)
    # startPos = (1.0, 1.0)
    # print 'startPos', startPos
    if len(invaders) > 0:
      features['distToHome'] = 0
    else:
      features['distToHome'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(startPos, myPos)
      # features['distToHome'] = self.getMazeDistance(self.startPosition,myPos)
      # features['distToHome'] = self.getMazeDistance(startPos, myPos)
      # features['distToHome'] = self.getMazeDistance(self.start, myPos)
   
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToHome': 10}
    # return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToHome': 3}

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

class MState():
  NUM_TURNS = 5
  GOAL = 0

  def __init__(self, gameState, index, value=0, move = None, fromMove = None, turn=NUM_TURNS):  
    self.gameState=gameState
    self.index=index
    self.value=value
    self.turn=turn
    self.move=move
    self.fromMove=fromMove

  def setMState(self, gameState, index):
    self.gameState=gameState
    self.index=index

  def resetMState(self, gameState, index, value):
    self.gameState=gameState
    self.index=index
    self.value=value

  def deepCopy( self ):
    mstate = MState( self, self.gameState, self.index )
    mstate.gameState = self.gameState
    mstate.index = self.index
    mstate.value = self.value
    mstate.turn = self.turn
    mstate.move = self.move
    mstate.fromMove = self.fromMove
    return mstate

  def next_mstate(self):

    # print '|||||||next_mstate'
    # print '-------gameState', self.gameState.getAgentState(self.index).getPosition()
    # print '-------value', self.value

    actions = self.gameState.getLegalActions(self.index)
    # The agent should not stay STOP
    actions.remove(Directions.STOP)

    # print '+++++Legal actions', actions
    nextmove = random.choice([x for x in actions])
    self.move = nextmove

    nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
    nextValue = DefenseAgent(self.index, 0.1).evaluate(self.gameState, nextmove)

    nextMState = MState(nextGameState, self.index, nextValue, nextmove, nextmove, self.turn-1)

    # print '-------self.move', self.move
    # print '-------nextGameState', nextGameState.getAgentState(self.index).getPosition()
    # # print '-------nextValue', nextValue
    # print '-------nextState', nextMState
    
    return nextMState
    
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
    s="Value: %d; Move: %s"%(self.value,self.move)
    return s
  


class Node():
  def __init__(self, mstate, parent=None):
    self.visits=1
    self.reward=0.0 
    self.mstate=mstate
    self.children=[]
    self.parent=parent  
  def add_child(self,child_mstate):
    child=Node(child_mstate,self)
    self.children.append(child)
  def getState(self):
    return self.mstate.deepCopy()
  def resetNode(self):
    self.visits=0
    self.reward=0.0 
    self.children=[]
  def update(self,reward):
    self.reward+=reward
    self.visits+=1
  def fully_expanded(self):
    # if len(self.children)==self.mstate.num_moves:
    actions = self.mstate.gameState.getLegalActions(self.mstate.index)
    actions.remove(Directions.STOP)
    availableActions = len(actions)
    print 'availableActions', availableActions, actions
    if len(self.children) == availableActions:  
      return True
    return False
  def __repr__(self):
    s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
    return s
    

def UCTSEARCH(budget,root,index):
  print 'UCTSEARCH'
  # print 'location', root.mstate.gameState.getAgentPosition(index)
  for iter in range(budget):
    if iter%10000==9999:
      logger.info("simulation: %d"%iter)
      logger.info(root)
    print 'root', root, 'type(root)', type(root)
    front=TREEPOLICY(root,index)
    reward=DEFAULTPOLICY(front.mstate)
    BACKUP(root,front,reward)
  # print 'root location', root.mstate.gameState.getAgentPosition(index)
  # print 'c location', BESTCHILD(root,0,index).mstate.gameState.getAgentPosition(index)
  return BESTCHILD(root,0,index)

def TREEPOLICY(node,index):
  print 'TREEPOLICY'
  # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^root before if', node, 'type(root)', type(node)

  if type(node) is list:  
    print 'node', node
    while node.getState.terminal()==False:
      # print 'terminal()==False'
      if node[0].fully_expanded()==False:  
        # print 'fully_expanded()==False'
        return EXPAND(node[0])
      else:
        node[0]=BESTCHILD(node[0],SCALAR,index)
    return node
  else:
    # print 'node', node
    # print 'node.mstate', node.getState()
    # print 'node.mstate.turn', node.getState().turn
    # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^root', node, 'type(root)', type(node)
    while node.getState().terminal()==False:
      # print 'terminal()==False'
      if node.fully_expanded()==False:  
        # print 'fully_expanded()==False'
        return EXPAND(node)
      else:
        print 'Fully Expanded'
        node=BESTCHILD(node,SCALAR,index)
    return node

def EXPAND(node):
  print 'EXPAND'
  tried_children_mstate=[c.mstate for c in node.children]
  print 'tried_children_state', tried_children_mstate
  new_mstate=node.mstate.next_mstate()
  # print 'new_mstate', new_mstate
  counter = 0
  while new_mstate in tried_children_mstate:
    actions = node.mstate.gameState.getLegalActions(node.mstate.index)
    actions.remove(Directions.STOP)
    print 'new_mstate in tried, len(actions):', len(actions)
    # print 'node location', node.mstate.gameState.getAgentState(node.mstate.index).getPosition()
    if len(actions) == 1:   # if it only has one possible direction, then jump out
      break
    new_mstate=node.mstate.next_mstate()
    counter = counter+1   # still not solved
    if counter > 5:
      break
    # print 'new_mstate', new_mstate   
  print 'Until new_mstate not in tried_children_mstate', new_mstate
  node.add_child(new_mstate)
  # for child in node.children:
  #   print 'node.children.mstate', child.mstate
  # print 'new_mstate in tried_children_mstate', new_mstate
  return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar,index):
  print 'BESTCHILD'
  print 'node location', node.mstate.gameState.getAgentPosition(index)
  bestscore = -9999999999
  bestchildren=[]
  for c in node.children:
    print 'c location', c.mstate.gameState.getAgentPosition(index)
    exploit=c.reward/c.visits
    explore=math.sqrt(math.log(2*node.visits)/float(c.visits))  
    score=exploit+scalar*explore
    print 'score', score, 'bestscore', bestscore
    if score==bestscore:
      bestchildren.append(c)  # more than one best child
    if score>bestscore:       # best child
      bestchildren=[c]
      bestscore=score
  
  # print 'find all children, bestchildren=', bestchildren
  if len(bestchildren)==0:
    logger.warn("OOPS: no best child found, probably fatal")

  # print 'index', index
  # if bestchildren!=[] and index==2:
  if bestchildren!=[]:
    # return random.choice(bestchildren)
    print 'bestchildren', bestchildren
    return bestchildren[0]
  else:
    return bestchildren

def DEFAULTPOLICY(mstate):
  print 'DEFAULTPOLICY'
  while mstate.terminal()==False:
    # print 'mstate.terminal()==False'
    mstate=mstate.next_mstate()
  return mstate.reward()

def BACKUP(root,node,reward):
  print 'BACKUP'
  while node!=None:
    node.visits+=1
    node.reward+=reward
    node=node.parent
    # node.parent = root
    # if node!=None:
      # print 'node.parent.mstate', node.mstate
  return


