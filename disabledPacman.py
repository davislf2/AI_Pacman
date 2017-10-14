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
import sys
sys.path.append('teams/disabledPacman')
from game import Directions, Agent, Actions
from captureAgents import CaptureAgent, AgentFactory
import random,util,time,distanceCalculator 



class reinforcementFactory(AgentFactory):
    def __init__(self, isRed):
        AgentFactory.__int__(self,isRed)
        self.agentList = ['MultiPurposeAgent','MultiPurposeAgent']

    def getAgent(self,index):
        if len(self.agentList) > 0:
            agent = self.agentList.pop(0)
            if agent == 'MultiPurposeAgent':
                return MultiPurposeAgent(index)
            return MultiPurposeAgent(index)

def createTeam(firstIndex, secondIndex, isRed,first = 'MultiPurposeAgent', second = 'MultiPurposeAgent'):
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
                'distance-to-mate': 0.18, 'distance-to-capsule': -30, 'eat-capsule':200, 'eat-scared-ghost': 200}

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
            # being close to teammate (negative reward (tiny))
            features['distance-to-mate'] = self.getMazeDistance(newPos, state.getAgentPosition(teams[0]))
            # get number of ghosts
            ghosts = []
            ghostState = []
            opp0 = state.getAgentState(opponents[0])
            opp1 = state.getAgentState(opponents[1])
            if not opp0.isPacman:
                ghosts.append(state.getAgentPosition(opponents[0]))
                ghostState.append(opp0)
            if not opp1.isPacman:
                ghosts.append(state.getAgentPosition(opponents[1]))
                ghostState.append(opp1)
            features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            # if action, eats scared ghost, flags it (with 1 and correspond to high weight)

            if opp0.scaredTimer > 0:
                if (next_x,next_y) == opponents[0]:
                    features['eat-scared-ghost'] = 1
            
            if opp1.scaredTimer > 0:
                if (next_x,next_y) == opponents[1]:
                    features['eat-scared-ghost'] = 1
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
