#
# This agent combines Learning agent from project 3 in an attempt
# Approximate Q-learning agent.
# 
# The reward function is defined, but the weights are not learnt,
# so this agent is actually running on heuristics 
# 
# Feature vector is modified below 
# 
# 
# These are modified codes from Part 3 for training the agent


# import sys
# sys.path.append('teams/disabledPacman')
# from game import Directions, Agent, Actions
# from captureAgents import CaptureAgent, AgentFactory
# import random,util,time,distanceCalculator 


import sys
sys.path.append('teams/disabledPacman')
from game import Directions, Agent, Actions
from captureAgents import CaptureAgent, AgentFactory
import random,util,time,distanceCalculator 
import game
from util import nearestPoint
from util import pause
from capture import noisyDistance
import math
import hashlib
import logging
import argparse


######################
# Parameters of MCTS #
######################

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)
NUM_SIM=10    # number of simulation times, at lease 1
REWARD_DISCOUNT=0.2
# REWARD_DISCOUNT=0   # if don't want to use roll out, just turn reward discount into 0
SIM_LEVEL=3   # level of tree expanding
LEVEL=0       # level of simulation tree (not used)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class reinforcementFactory(AgentFactory):
    def __init__(self, isRed):
        AgentFactory.__int__(self,isRed)
        self.agentList = ['MultiPurposeAgent','DefenseAgent']

    def getAgent(self,index):
        if len(self.agentList) > 0:
            agent = self.agentList.pop(0)
            if agent == 'MultiPurposeAgent':
                return MultiPurposeAgent(index)
            return DefenseAgent(index)

def createTeam(firstIndex, secondIndex, isRed,first = 'MultiPurposeAgent', second = 'DefenseAgent'):
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
  
  return [eval(first)(index = firstIndex,isRed = isRed),
          eval(second)(index = secondIndex,isRed = isRed)]


class MultiPurposeAgent(CaptureAgent):
    """
    Subclass of CaptureAgent, but contains components for capture.py 
    to call during training / game.run() 

    Initially designed to run as Approximate Q-learning agent,
    with reward function being hard to define to converge properly,
    the weights are given manually, and this agent is now considered 
    heuristics search agent

    Function used for the training are commented out (final)
    some other are left alone since it will not affect the game

    Reward function and update function is still called by the game
    through observationFunction, but these are irrelevant, since weights
    are predefined using heuristics
    """
    def __init__(self, index, isRed,actionFn = None,numTraining = 0, 
                 epsilon = 0.0, alpha =0.2, discount=0.8):
        # initialize superclass
        CaptureAgent.__init__(self,index)
        # parameters for training
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions(index)
        self.actionFn = actionFn
        # these are for reinforcement (since it's become heuristic search)
        # I just leave them for further improvement later on
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = epsilon 
        self.alpha= alpha 
        self.discount= discount 
        self.weights = util.Counter()

        self.visitedPositions = {}

        self.red = isRed

        # print("i'm on RED team?",self.red,"with index",self.index)
        # print("[DEBUG] Agent initialized")

    def registerInitialState(self,gameState):
        self.startEpisode()
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self,gameState)
        # print("registering intial state")


        # register weight here
        # Competition would require offline training prior
    
    ########################################################
    ################## GETTER METHODS #######################
    ### structure defined same way as material for proj 3 ###

    def getLegalActions(self,state):
        return self.actionFn(state)

    def getQValue(self, state, action):
        """
        return Qvalue for the action
        for this implementation, it is heuristics, not learnt weight that 
        gives the Q value 
        
        commented out version can learn the weight through reinforcment,
        but the value does not converge easily
        """
        agentState = state.getAgentState(self.index)
        featureVector = self.getFeatures(state,action,self.index,self.red, agentState)
        Qval = 0
        
        # for key in featureVector:
            # Qval += self.weights[key] * featureVector[key]
        # print "Qval is ", Qval
        weights = self.getManualWeights()

        for key in featureVector:
        #     print "key used to computed [" ,key,"] value [", featureVector[key],"]"
            Qval += weights[key] * featureVector[key]
        ### using manual weight version 
        return Qval

    def getManualWeights(self):
        
        return {'closest-food':-100, 'eats-food':100,'bias':3,'#-of-ghosts-1-step-away':-500,
                'distance-to-base':-100, 'distance-to-eat': -40, 'eat-pacman':200, 'avoid-pacman-power':-500,
                'distance-to-capsule': -30, 'eat-capsule':200, 'eat-scared-ghost': 200}

    def getFeatures(self, state, action, index, red, agentState):
        """
        built on top of project 3 codes of simple features extractor
        same features (ghosts 1 step away) from project 3 codes is use without much modification
        """
        ## All relevants information for the features##
        try:
            walls = state.getWalls()
            pos = state.getAgentPosition(index)
            x = pos[0]
            y = pos[1]
            # print "[DEBUG] agent position at" , pos
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            # print "[DEBUG] Agent", index ,"team", red, "next move to ", next_x,next_y
            food = self.getFood(state)
            myfood = self.getFoodYouAreDefending(state)
            newPos =(next_x,next_y)
            distToFood = self.getClosestFood(newPos,food.asList())
            opponents = self.getOpponents(state)
            teams = self.getTeam(state)

            features = util.Counter()

            teams.remove(self.index)
            # get number of ghosts
            ghosts = []
            ghostState = []
            opp0 = state.getAgentState(opponents[0])
            opp1 = state.getAgentState(opponents[1])
            if not opp0.isPacman:
                if opp0.scaredTimer > 0 and (next_x,next_y) == state.getAgentPosition(opponents[0]):
                    features['eat-scared-ghost'] = 1
                else:
                    ghosts.append(state.getAgentPosition(opponents[0]))
                    ghostState.append(opp0)
            if not opp1.isPacman:
                if opp1.scaredTimer > 0 and (next_x,next_y) == state.getAgentPosition(opponents[1]):
                    features['eat-scared-ghost'] = 1
                else:
                    ghosts.append(state.getAgentPosition(opponents[1]))
                    ghostState.append(opp1)
            features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            # if action, eats scared ghost, flags it (with 1 and correspond to high weight)

            # if opp0.scaredTimer > 0:
            #     if (next_x,next_y) == opponents[0]:
            #         features['eat-scared-ghost'] = 1
            
            # if opp1.scaredTimer > 0:
            #     if (next_x,next_y) == opponents[1]:
            #         features['eat-scared-ghost'] = 1
            # if action gets you to ghosts, flags it (with 1 and correspond to negative weight)
            features['bias'] = 1
            if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
                features["eats-food"] = 1.0

            if distToFood is not None:
                features["closest-food"] = float(distToFood) / (walls.width * walls.height)

            nextState = state.generateSuccessor(self.index,action)
            nextAgentState = nextState.getAgentState(self.index)
            
            if agentState.numCarrying > 3:
                xbase = walls.width / 2
                ybase = walls.height
                if red:
                    distTobase = self.getMazeDistance(self.start,newPos)
                    features['distance-to-base'] = distTobase
                else:
                    distTobase = self.getMazeDistance(self.start,newPos)
                    features['distance-to-base'] = distTobase
            
            if agentState.numCarrying >= 1:
                if min(self.getMazeDistance(pos,ghosts[0]), self.getMazeDistance(pos,ghosts[1])) <= 5:
                    xbase = walls.width / 2
                    ybase = walls.height
                    if red:
                        distTobase = self.getMazeDistance(self.start,newPos)
                        features['distance-to-base'] = distTobase
                    else:
                        distTobase = self.getMazeDistance(self.start,newPos)
                        features['distance-to-base'] = distTobase

            pacman = []
            if opp0.isPacman: 
                pacman.append(state.getAgentPosition(opponents[0]))
            if opp1.isPacman:
                pacman.append(state.getAgentPosition(opponents[1]))
    
        except Exception:
            pass
            pacman = []

        #### Features relevants to pacman in the area
        try:
            if agentState.scaredTimer == 0:
                distToEat0 = self.getMazeDistance(newPos,pacman[0])
        except Exception:
            distToEat0 = 999
        try:
            if agentState.scaredTimer == 0:
                distToEat1 = self.getMazeDistance(newPos,pacman[1]) 
        except Exception:
            distToEat1 = 999

        try:
            # if found pacman, give distance, (chase them)
            distToEat = min(distToEat0,distToEat1)
            if distToEat != 999:
                features['distance-to-eat'] = distToEat    

            capsules = self.getCapsules(state)
            if (next_x,next_y) in capsules:
                features['eat-capsule'] = 1 

        except Exception:
            pass
        try:
            if len(pacman) > 0:
                if agentState.scaredTimer > 0:
                    if pacman[0] == (next_x,next_y):
                        features['avoid-pacman-power'] = 1
                    if pacman[1] == (next_x,next_y):
                        features['avoid-pacman-power'] = 1
                else:
                    if pacman[0] == (next_x,next_y):
                        features['eat-pacman'] = 1
                    if pacman[1] == (next_x,next_y):
                        features['eat-pacman'] = 1
        except IndexError:
            pass
        
        features.divideAll(10.0)

        return features

    def getClosestFood(self,position, foodList):
        try:
            return min(list(map( lambda x: self.getMazeDistance(x,position),foodList)))
        except ValueError:
            return None

    def getReward(self,state,index):
        """
        Reward function, defining reward for the game
        
        This function is called during the game, but the weights are not learnt by the agent
        Weight are given to the agents as heuristics instead

        """
        score = state.getScore() 
        wall = state.getWalls()
        rfood = state.getRedFood()
        bfood = state.getBlueFood()
        x,y = state.getAgentPosition(index)
        distToFood = closestFood((x,y),bfood,wall)
        try:
            distToFood = float(distToFood) / (wall.width * wall.height)
        except TypeError:
            distToFood = None

        agentState = state.getAgentState(self.index).copy()
        newArea = 0
        if state.getAgentPosition(index) not in self.visitedPositions:
            newArea = 10 
            self.visitedPositions[state.getAgentPosition(index)] = True 
        scored = agentState.numReturned
        carry = agentState.numCarrying
        reward = (10*scored**2) + carry +  100*score + newArea 
        # print "[REWARD] ", reward
        # util.pause()
        return reward


    def chooseAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions) == 0:
            ## update prior to sending back action
            self.lastState = state
            self.lastAction = action
            return None
        if util.flipCoin(self.epsilon):
            # print "take random exploration"
            self.lastState = state
            action = random.choice(legalActions)
            self.lastAction = action
            return action
        self.lastState = state
        action = self.computeActionFromQValues(state) 
        self.lastAction = action
        # print "taking action" , action
        return action 

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
        # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


    ########################################
    # Function RELEVANT to Pacman TRAINING #
    # Some are copied from QlearningAgent #
    # since the capture.py is similar to pacman.py #
    # it's possible to use them

    def update(self,state,action,nextState,reward):
        """
        Update the weights for A
        """
        # print "updating"
        difference = reward + (
            self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)
        featureVector = self.getFeatures(state,action,self.index,self.red,state.getAgentState(self.index))
        for key in featureVector:
            self.weights[key] += self.alpha * difference * featureVector[key]
        # print "weight Updated to" , self.weights

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActs = self.getLegalActions(state)
        if len(legalActs) == 0:
            return 0.0
        values = []
        for action in legalActs:
            values.append(self.getQValue(state,action))
        return max(values)
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActs = self.getLegalActions(state)
        # print "LEGAL ACTION INCLUDE", legalActs
        if len(legalActs) == 0:
            return None
        best_action = ''
        bestQval = float("-inf")
        for action in legalActs:
            # print "[DECISION] Agent",self.index, ", Action", action, "to be taken has Qvalue",
            #  "--> [[[ ", self.getQValue(state,action), "]]]"

            if self.getQValue(state,action) >= bestQval:
                bestQval = self.getQValue(state,action)
                best_action = action
        # print "[DEBUG] Agent",self.index,"computed the best action to be", best_action
        # print "[DEBUG] with Qval," , bestQva
        return best_action

    # observationFunction gets called by the game.run()
    # it updates the simulation
    # reward function is called here defined here
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = self.getReward(state,self.index)
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state


    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Same function from Project 3, the capture.py looks for 
            and execute this function, so this has to remain unchanged

            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        # episodeRewards initiized by start Episode below
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

# Final method is taken off to keep code brief

def closestFood(pos, food, walls):
    """
    closestFood -- this method is defined in the reinforcement learning 
    project provided by Berkeley
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None



##############
# MCTSAgents #
##############

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
    #print '++++++++++++++++attacker eval time for agent %d: %.4f' % (self.index, time.time() - start)

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

######################
# Normal AttackAgent #
######################

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
      #print 'Ghost Positions:', positions  # Ghost index: 1, 3
      #print 'Pacman Position:', myPos     # Pacman index 0
      #print 'distanceToGhost', features['distanceToGhost']     
    else:
      ## print 'noisyDistanceToGhost1', dist1, 'noisyDistanceToGhost1', dist2
      features['distanceToGhost'] = 0
      #print 'no ghost'

    # Compute distance to the nearest Power Pills
    # feature['distanceToPower']
    
    ## print self.getPreviousObservation()

    return features

  def getWeights(self, gameState, action):

    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 10}

#####################
# MCTS DefenseAgent #
#####################

class DefenseAgent(MCTSAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def __init__(self, index, isRed,actionFn = None,numTraining = 0, 
               epsilon = 0.0, alpha =0.2, discount=0.8):
    # initialize superclass
    CaptureAgent.__init__(self,index)
    # parameters for training
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions(index)
    self.actionFn = actionFn
    # these are for reinforcement (since it's become heuristic search)
    # I just leave them for further improvement later on
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = epsilon 
    self.alpha= alpha 
    self.discount= discount 
    self.weights = util.Counter()

    self.visitedPositions = {}

    self.red = isRed


#######  Monte Carlo Tree Search Simulation
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
    
    self.numSims=NUM_SIM
    self.sturns=SIM_LEVEL
    self.levels=LEVEL
    self.svalue=0 
    self.smoves=[]
    self.gameState = gameState
    self.current_node = Node( MState(self.gameState, self.index, self.svalue, self.smoves, self.sturns) )
    #print '~~~register self.current_node', self.current_node
    self.startPosition = self.current_node.mstate.gameState.getAgentState(self.index).getPosition()
    #print '~~~register Start location', self.startPosition

  def getDefAction(self, gameState):
    """
    Prevent CaptureAgent always use the overriden chooseAction from DefenseAgent
    """
    self.observationHistory.append(gameState)

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    if myPos != nearestPoint(myPos):
      # We're halfway from one position to the next
      return gameState.getLegalActions(self.index)[0]
    else:
      return self.chooseDefAction(gameState)

  def chooseDefAction(self, gameState):

    """
    Picks among the actions with the highest Q(s,a).
    """
    return self.runSimulation(self.current_node, gameState, self.index, self.levels, self.numSims)

  def runSimulation(self, current_node, gameState, index, levels=3, numSims=5):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # pause()
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    

    #print 'runSim/////////////////////////////////////////////'
    self.gameState = gameState
    self.index = index  
    self.current_node = current_node

    value = 0
    self.current_node.resetNode()
    self.current_node.mstate.resetMState(self.gameState, index, value)

    ## print 'self.current_node', self.current_node, 'type', type(self.current_node)
    #print 'Current Location', self.current_node.mstate.gameState.getAgentState(self.index).getPosition(), 'type', type(self.current_node.mstate.gameState.getAgentState(self.index).getPosition())
    ### This location move!!
    self.index = index

    self.current_node.children = []

    l = LEVEL
    child_node=UCTSEARCH(numSims/(l+1), self.current_node, self.index)
    #print 'Child Location', child_node.mstate.gameState.getAgentState(self.index).getPosition()
    #print 'Child_node.mstate.fromMove', child_node.mstate.fromMove

    #print '++++++++++++++++defender eval time for agent %d: %.4f' % (self.index, time.time() - start)

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

  def getIsRed(self):
    if self.index%2 == 0:
      return True
    else:
      return False

  def getFeatures(self, gameState, action):

    self.gameState = gameState

    features = util.Counter()
    successor = self.getSuccessor(self.gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # self.isRed = False
    # if self.index%2 == 0:
    #   self.isRed = True
    # else:
    #   self.isRed = False

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # state = self.gameState.deepCopy()
    # Adds the sonar signal
    pos = successor.getAgentPosition(self.index)
    n = successor.getNumAgents()
    # print 'getNumAgents()', n # 4
    distances = []

    # for i in range(n):
    #   if successor.getAgentPosition(i) != None:
    #     distances.append(noisyDistance(pos, successor.getAgentPosition(i)))

    # print 'distances', distances

    # distances = [noisyDistance(pos, state.getAgentPosition(i)) for i in range(n)]
    
    # state.agentDistances = distances

    # gState = self.gameState.makeObservation(self.index)
    
    dists = []

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    # Invader: Enemy in the vision
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      # dists = [self.gameState.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # print 'invaders', invaders
      poss = [a.getPosition() for a in invaders]
      # print 'poss', poss
      # if self.distancer != None:
        # dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # dists = [super(DefenseAgent, self).getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # dists = [distanceCalculator.Distancer(self.gameState.data.layout).getDistance(myPos, a.getPosition()) for a in invaders]
      
      if dists != []:
        features['invaderDistance'] = min(dists)
        # print 'min(dists)', min(dists)
      else:
        features['invaderDistance'] = 0

      # for i in self.getOpponents(successor):
      #   nd = noisyDistance(pos, successor.getAgentPosition(i))
      #   if nd > 0:
      #     print 'noisyDist', nd
      #     dists.append(nd)

      # print 'dists', dists
      # features['invaderDistance'] = min(dists)

    # else:
      # features['invaderDistance'] = min(distances)

    # #  Enemy index: 1, 3
    # for i in invaders:
    #   print '******Enemy Pos:', i.getPosition(), 'found by index', self.index
    # print 'Index', self.index, 'Defender Pos:', myPos     #  index 0

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    # In the begining, leave home as far as possible
    ## print 'myPos', myPos
    startPos = gameState.getInitialAgentPosition(self.index)
    # startPos = (1.0, 1.0)
    ## print 'startPos', startPos
    if len(invaders) > 0:
      features['distToHome'] = 0
    else:
      features['distToHome'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(startPos, myPos)
      # features['distToHome'] = self.getMazeDistance(self.startPosition,myPos)
      # features['distToHome'] = self.getMazeDistance(startPos, myPos)
      # features['distToHome'] = self.getMazeDistance(self.start, myPos)


    if self.getIsRed():
      centralX = (gameState.data.layout.width - 2)/2
    else:
      centralX = ((gameState.data.layout.width - 2)/2) + 1

    centralY = (gameState.data.layout.height)/2
    centralPos = (centralX, centralY)

    if len(invaders) > 0:
      features['distToCentral'] = 0
    else:
      features['distToCentral'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(centralPos, myPos)
    

    features['distToHome'] = 0
    features['distToCentral'] = 0

    return features


  def getWeights(self, gameState, action):
    # return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToCentral': -3}
    # return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distToHome': 5, 'distToCentral': -5}
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


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

  def deepCopy(self):
    mstate = MState( self, self.gameState, self.index )
    mstate.gameState = self.gameState
    mstate.index = self.index
    mstate.value = self.value
    mstate.turn = self.turn
    mstate.move = self.move
    mstate.fromMove = self.fromMove
    return mstate

  def getFromMove(self):
    return self.fromMove
  
  def getMove(self):
    return self.move

  # def next_mstate(self, oMState = None):
  def next_mstate(self, oNode = None):

    ## print '|||||||next_mstate'
    ## print '-------gameState', self.gameState.getAgentState(self.index).getPosition()
    ## print '-------value', self.value

    actions = self.gameState.getLegalActions(self.index)
    # The agent should not stay STOP
    actions.remove(Directions.STOP)

    # if oMState == None:
    #   otherMState = []
    # else:
    #   otherMState = oMState
    ##   print 'otherMState', otherMState
    #   if otherMState != []:
    #     actions.remove(otherMState.getFromMove())   

    # remove have tried children moves from actions
    if oNode == None:
      otherNode = []
    else:
      otherNode = oNode
      #print 'otherNode', otherNode
      if otherNode != []:
        for n in otherNode:
          # print 'tried_children.mstate.getFromMove()', n.mstate.getFromMove()
          actions.remove(n.mstate.getFromMove())  
          # print 'tried_children.mstate.getMove()', n.mstate.getMove()
          # actions.remove(n.mstate.getMove())  

    da = DefenseAgent(self.index, 0.1)
    da.registerInitialState(self.gameState.deepCopy())
    # da.setPosition(self.gameState.getAgentState(self.index).getPosition())
    
    nextValues = [da.evaluate(self.gameState, a) for a in actions]
    # nextValue = da.evaluate(self.gameState, nextmove)

    nextValue = max(nextValues)
    bestActions = [a for a, v in zip(actions, nextValues) if v == nextValue]

    ## print '+++++Legal actions', actions
    nextmove = random.choice([x for x in bestActions])
    self.move = nextmove

    nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
    # nextValue = DefenseAgent(self.index, 0.1).evaluate(self.gameState, nextmove)



    # if da.getIsRed():
    ##   print '~~~~~~~~~~~~ red index', self.index # Red
    # else:
    ##   print '~~~~~~~~~~~~ blue index', self.index

    # nextFeatures = DefenseAgent(self.index, 0.1).getFeatures(self.gameState, nextmove)
    ## print '***************** action', nextmove
    ## print 'Features', nextFeatures

    # nextMState = MState(nextGameState, self.index, nextValue, nextmove, nextmove, self.turn-1)
    nextMState = MState(nextGameState, self.index, nextValue, None, nextmove, self.turn-1)

    ## print '-------self.move', self.move
    ## print '-------nextGameState', nextGameState.getAgentState(self.index).getPosition()
    ## # print '-------nextValue', nextValue
    ## print '-------nextState', nextMState
    
    return nextMState
    
  def terminal(self):
    if self.turn == 0:
      return True
    return False
  def reward(self):
    # r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
    r = self.value
    return r
  # def __hash__(self):
  #   # return int(hashlib.md5(str(self.moves)).hexdigest(),16)
  #   return int(hashlib.md5(str(self.move)).hexdigest(),16)
  # def __eq__(self,other):
  #   if hash(self)==hash(other):
  #     return True
  #   return False
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
  def setParentNode(self, node):
    self.parent = node
  def update(self,reward):
    self.reward+=reward
    self.visits+=1
  def fully_expanded(self):
    # if len(self.children)==self.mstate.num_moves:
    actions = self.mstate.gameState.getLegalActions(self.mstate.index)
    actions.remove(Directions.STOP)
    availableActions = len(actions)
    #print 'availableActions', availableActions, actions
    #print 'len(self.children)', len(self.children), self.children
    if len(self.children) == availableActions:   # If it's fully expanded
      #printChildrenPosition(self.children)
      return True
    return False
  def __repr__(self):
    s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
    return s

# For debugging
# def printChildrenPosition(children):
#   if len(children) == 1:
#     #print 'Children Position == 1'
#     #print children[0].getState().gameState.getAgentPosition(2)
#   elif len(children) > 1:
#     #print 'Children Position > 1'
#     for c in children:
#       #print c.getState().gameState.getAgentPosition(2)

# def printChildrenStatePosition(children):
#   if len(children) == 1:
#     #print 'Children State Position == 1'
#     #print 'children', children
#     #print children.gameState.getAgentPosition(2)
#   elif len(children) > 1:
#     #print 'Children State Position > 1'
#     for c in children:
#       #print c.gameState.getAgentPosition(2)

def UCTSEARCH(budget,root,index):
  #print 'UCTSEARCH'
  ## print 'location', root.mstate.gameState.getAgentPosition(index)
  for iter in range(budget):
    if iter%10000==9999:
      logger.info("simulation: %d"%iter)
      logger.info(root)
    ## print 'root', root, 'type(root)', type(root)
    front=TREEPOLICY(root,index)
    reward=DEFAULTPOLICY(front.mstate)
    BACKUP(root,front,reward)
  ## print 'root location', root.mstate.gameState.getAgentPosition(index)
  ## print 'c location', BESTCHILD(root,0,index).mstate.gameState.getAgentPosition(index)
  return BESTCHILD(root,0,index)

def TREEPOLICY(node,index):
  #print 'TREEPOLICY'
  ## print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^root before if', node, 'type(root)', type(node)

  if type(node) is list:  
    #print 'node', node
    while node.getState.terminal()==False:
      ## print 'terminal()==False'
      if node[0].fully_expanded()==False:  
        ## print 'fully_expanded()==False'
        return EXPAND(node[0])
      else:
        node[0]=BESTCHILD(node[0],SCALAR,index)
    return node
  else:
    ## print 'node', node
    ## print 'node.mstate', node.getState()
    ## print 'node.mstate.turn', node.getState().turn
    ## print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^root', node, 'type(root)', type(node)
    while node.getState().terminal()==False:
      ## print 'terminal()==False'
      if node.fully_expanded()==False:  
        ## print 'fully_expanded()==False'
        return EXPAND(node)
      else:
        #print 'Fully Expanded'
        node=BESTCHILD(node,SCALAR,index)
    return node

def EXPAND(node):
  #print 'EXPAND'
  # tried_children_mstate=[c.mstate for c in node.children]
  # if tried_children_mstate != []:
  ##   print 'tried_children', tried_children_mstate
  ##   print 'tried_children_mstate', tried_children_mstate.mstate
  ## print 'tried_children_state position', printChildrenStatePosition(tried_children_mstate)


  tried_children=[c for c in node.children]

  # if tried_children != []:
    #print 'tried_children', tried_children
    ## print 'tried_children.mstate', tried_children.mstate
    #print 'tried_children position', printChildrenPosition(tried_children)  

  # tried_children_mstate.gameState.getAgentPosition(2)  
  # new_mstate=node.mstate.next_mstate(tried_children_mstate)

  new_mstate=node.mstate.next_mstate(tried_children)

  tried_children_mstate = []

  if tried_children != []:
    for t in tried_children:
      tried_children_mstate.append(t.mstate)
  # else:
  #   tried_children_mstate = []

  # if tried_children_mstate != []:
  ##   # print 'new_mstate', new_mstate
  #   # Then it should not go here
  #   counter = 0
  #   while new_mstate in tried_children_mstate:
  #     actions = node.mstate.gameState.getLegalActions(node.mstate.index)
  #     actions.remove(Directions.STOP)
  ##     print 'new_mstate in tried, len(actions):', len(actions)
  ##     # print 'node location', node.mstate.gameState.getAgentState(node.mstate.index).getPosition()
  #     if len(actions) == 1:   # if it only has one possible direction, then jump out
  #       break
  #     new_mstate=node.mstate.next_mstate()
  #     counter = counter+1   # still not solved
  #     if counter > 5:
  #       break
  ##     # print 'new_mstate', new_mstate   


  #print 'Until new_mstate not in tried_children_mstate', new_mstate
  node.add_child(new_mstate)
  node.children[-1].setParentNode(node)

  # for child in node.children:
  ##   print 'node.children.mstate', child.mstate
  ## print 'new_mstate in tried_children_mstate', new_mstate
  return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar,index):
  #print 'BESTCHILD scalar  ====================  ', scalar
  #print 'node location', node.mstate.gameState.getAgentPosition(index)
  bestscore = -9999999999
  bestchildren=[]
  for c in node.children:
    #print 'c location', c.mstate.gameState.getAgentPosition(index)
    exploit=c.reward/c.visits
    explore=math.sqrt(math.log(2*node.visits)/float(c.visits))  
    score=exploit+scalar*explore
    #print 'score', score, 'bestscore', bestscore
    if score==bestscore:
      bestchildren.append(c)  # more than one best child
    if score>bestscore:       # best child
      bestchildren=[c]
      bestscore=score
  
  ## print 'find all children, bestchildren=', bestchildren
  if len(bestchildren)==0:
    logger.warn("OOPS: no best child found, probably fatal")

  ## print 'index', index
  # if bestchildren!=[] and index==2:
  if bestchildren!=[]:
    return random.choice(bestchildren)
    #print 'bestchildren ---------------------- ', bestchildren[0].mstate.gameState.getAgentPosition(index)
    # return bestchildren[0]
  else:
    return bestchildren

def DEFAULTPOLICY(mstate):
  #print 'DEFAULTPOLICY'
  while mstate.terminal()==False:
    ## print 'mstate.terminal()==False'
    mstate=mstate.next_mstate()
  return mstate.reward()

def BACKUP(root,node,reward):
  #print 'BACKUP'
  while node!=None:
    node.visits+=1
    node.reward+=reward*(REWARD_DISCOUNT**SIM_LEVEL) # discounted reward after several turns of simulation
    node.reward+=node.mstate.reward() # add the root reward and the last reward together

    node=node.parent
    # node.parent = root
    # if node!=None:
      ## print 'node.parent.mstate', node.mstate
  return 0

