#
# This agent combines Learning agent from project 3 to make
# Approximate Q-learning agent.
# Feature vector is modified below 
# 
# These are modified codes from Part 3 for training the agent
from game import Directions, Agent, Actions
from captureAgents import CaptureAgent
import random,util,time,distanceCalculator 

def createTeam(firstIndex, secondIndex, isRed,
               numTraining = 0,first = 'RLAgent', second = 'RLAgent'):
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
  
  # SOLVE this

#   print("first eval",eval(first)(firstIndex))
#   print("second eval", eval(second)(secondIndex))
  return [eval(first)(index = firstIndex,isRed = isRed, numTraining = numTraining),
          eval(second)(index = secondIndex,isRed = isRed, numTraining = numTraining)]


class RLAgent(CaptureAgent):
    """
    Subclass of CaptureAgent, but contains components for capture.py 
    to call during training / game.run() 
    

    Intended to use as offline-trained agent when in competition.
    Parameters initialized to be in training mode.
    These are set off to 0 when in testing mode
    """
    def __init__(self, index, isRed,actionFn = None,numTraining = 0, 
                 epsilon = 0.05, alpha =0.2, discount=0.8):
        # initialize superclass
        CaptureAgent.__init__(self,index)
        ##### Debug zone
        # print "TRAINING", numTraining
        # print "MY INDEX IS", self.index
        #####

        # parameters for training
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions(index)
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = epsilon 
        self.alpha= alpha 
        self.discount= discount 
        self.weights = util.Counter()

        self.red = isRed

        print("i'm on RED team?",self.red)
        print("[DEBUG] Agent initialized")

    def registerInitialState(self,gameState):
        self.startEpisode()
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self,gameState)
        print("registering intial state")


        # register weight here
        # Competition would require offline training prior
    
    ########################################################
    ################## GETTER METHODS #######################
    ### structure defined same way as material for proj 3 ###

    def getLegalActions(self,state):
        return self.actionFn(state)

    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        agentState = state.getAgentState
        featureVector = self.getFeatures(state,action,self.index,self.red, agentState)
        Qval = 0
        for key in featureVector:
            Qval += self.weights[key] * featureVector[key]
        return Qval

    def getFeatures(self, state, action, index, red, agentState):
        wall = state.getWalls()


        pos = state.getAgentPosition(index)
        x = pos[0]
        y = pos[1]
        # print "[DEBUG] agent position at" , pos
        # distToFood = closestFood((x,y),bfood,wall)
        food = self.getFood(state)
        myfood = self.getFoodYouAreDefending(state)

        # print "food returned", food.asList.()

        distToFood = self.getClosestFood(pos,food.asList())
        # print "closest food is " ,distToFood

        print "agent distance :", state.getAgentDistances()


        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # opponents = self.getOpponents()


        # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        
        # define features here to profit
        features = util.Counter()
        # below is the fake features I put in for now
        features['bias'] = 1

        return features

    def getClosestFood(self,position, foodList):
        return min(list(map( lambda x: self.getMazeDistance(x,position),foodList)))




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
            self.lastState = state
            action = random.choice(legalActions)
            self.lastAction = action
            return action
        self.lastState = state
        action = self.computeActionFromQValues(state) 
        self.lastAction = action
        return action 

    ########################################
    # Function RELEVANT to Pacman TRAINING #
    # These are copied from QlearningAgent #
    # since the capture.py is similar to pacman.py #
    # it's possible to use them

    def update(self,state,action,nextState,reward):
        """
        Update the weights for A
        """
        difference = reward + (
            self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)
        featureVector = self.getFeatures(state,action,self.index,self.red,state.getAgentState(self.index))
        for key in featureVector:
            self.weights[key] += self.alpha * difference * featureVector[key]
        print "weight Updated to" , self.weights

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
            if self.getQValue(state,action) >= bestQval:
                bestQval = self.getQValue(state,action)
                best_action = action
        # print "[DEBUG] computed the best action to be", best_action
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
            reward = self.getReward(state,self.index) - self.getReward(self.lastState,self.index)
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def getReward(self,state,index):
        """
        Reward function, defining reward for the game

        """
        score = state.getScore() 
        wall = state.getWalls()
        rfood = state.getRedFood()
        bfood = state.getBlueFood()
        x,y = state.getAgentPosition(index)
        distToFood = closestFood((x,y),bfood,wall)

        agentState = state.getAgentState(self.index).copy()
        
        scored = agentState.numReturned
        carry = agentState.numCarrying
        reward = (10*scored**2) + carry +  10*score + 0.2 * distToFood 
        return reward

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

    ### final method, called by game if found after each episode

    def final(self, state):
        """
          Called by Pacman game at the terminal state
          Same function as provided by the project 3
        """
        # it is working quite properly now,  
        try:
            deltaReward = state.getScore() - self.lastState.getScore()
        except Exception:
            deltaReward = 0
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print 'Reinforcement Learning Status:'
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print '\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining)
                print '\tAverage Rewards over all training: %.2f' % (
                        trainAvg)
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
                print '\tAverage Rewards over testing: %.2f' % testAvg
            print '\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg)
            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))

## Redefine this for efficiency
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
