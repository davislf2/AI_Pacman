"""
NOTE:
This file is based off example code provided by the instructors for AI planning
for Autonomy.

Our team did not use this for our competition planner. This unfinished file is
included to show how we developed and tested a basic PDDL Classical planning
solver.

Rather than using the FF solver locally, this planner uses the remote API
endpoint at planning.domains to generate plans, and as such, is very slow.
"""


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
import random, time, util
from game import Directions
import game

import os
from util import nearestPoint

import urllib2, json, sys
# bin_path = "../../bin"    # Windows
bin_path = "..\..\\bin"     # Mac or Unix


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ClassicalGhostAgent', second = 'ClassicalGhostAgent'):
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

# Base class that provides some utility functions for PDDL agents
class PDDLAgent(CaptureAgent):
    def __init__(self, index, timeForComputing = 0.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.visibleAgents = []

    # getLayoutDimensions
    # ===================
    # Returns the full dimensions of the game layout by checking the upper
    # right wall position
    def getLayoutDimensions(self, gameState):
        wallPositions = gameState.getWalls().asList(True)
        upperRightCorner = wallPositions[-1]
        (x, y) = upperRightCorner
        return (x, y)

    def findCentralLocation(self, gameState, sizeX, sizeY):
        positions = gameState.getWalls().asList(False)

        centralX = (sizeX / 2) + 1
        centralY = sizeY / 2

        if (centralX, centralY) in positions:
            return (centralX, centralY)

        step = 1
        while True:
            if (centralX, centralY + step) in positions:
                return (centralX, centralY + step)
            elif (centralX, centralY - step) in positions:
                return (centralX, centralY - step)
            else:
                step = step + 1

    def getPositionsForBlue(self, positions):
        bluePositions = [position for position in positions if position[0] > self.borderLine]
        return bluePositions

    def getPositionsForRed(self, positions):
        redPositions = [position for position in positions if position[0] < self.borderLine]
        return redPositions

class ClassicalGhostAgent(PDDLAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        (xSize, ySize) = self.getLayoutDimensions(gameState)
        self.borderLine = 1.0 * xSize / 2
        self.centralLocation = self.findCentralLocation(gameState, xSize, ySize)

        allPositions = gameState.getWalls().asList(False)
        if self.red:
            positions = self.getPositionsForRed(allPositions)
        else:
            positions = self.getPositionsForBlue(allPositions)

        self.objectsString = self.createPDDLObjects(positions)
        self.predicatesString = self.createPDDLPredicates(positions)

    def createPDDLObjects(self, positions):
        objects = list()
        objects.append("\t(:objects")

        for position in positions:
            (x, y) = position
            objects.append(" X%dY%d" % (x, y))

        objects.append(" - position)\n")

        return "".join(objects)

    def createPDDLPredicates(self, positions):
        predicates = list()
        predicates.append("\t(:init\n")

        for position in positions:
            (x, y) = position

            if (x - 1, y) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x - 1, y))
            if (x + 1, y) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x + 1, y))
            if (x, y - 1) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x, y - 1))
            if (x, y + 1) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x, y + 1))

        return "".join(predicates)

    # CURRENT GOAL IS THE NEAREST PACMAN
    def createPDDLGoal(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)

        # GET THE INDICES OF THE ENEMY AGENTS
        if self.red:
            enemies = gameState.getBlueTeamIndices()
        else:
            enemies = gameState.getRedTeamIndices()

        closestEnemyPosition = None
        closestEnemyDistance = 9999

        for enemy in enemies:
            position = gameState.getAgentPosition(enemy)
            if position != None and position[0] > self.borderLine:
                distance = self.distancer.getDistance(currentPosition, position)
                if distance < closestEnemyDistance:
                    closestEnemyPosition = position
                    closestEnemyDistance = distance

        if closestEnemyPosition != None:
            (goalX, goalY) = closestEnemyPosition
            return "\t\t(at X%dY%d)\n" % (goalX, goalY)


        # FILL THE CODE TO GENERATE PDDL GOAL
        # (goalX, goalY) = goalPosition
        (midX, midY) = self.centralLocation
        return "\t\t(at X%dY%d)\n" % (midX, midY)

    def generatePDDLProblem(self, gameState):
        # CONVERT A SCENARIO TO A STRIPS PROBLEM
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        # file = open("%s/ghostProblem%d.pddl" % (currentDirectory, self.index), "w") # Windosw
        file = open("%s\ghostProblem%d.pddl" % (currentDirectory, self.index), "w") # Mac or Unix

        lines = list()
        lines.append("(define (problem ghost%d)\n" % self.index)
        lines.append("\t(:domain ghost)\n")
        lines.append(self.objectsString)
        # lines.append(")\n")

        lines.append(self.predicatesString)

        (x, y) = gameState.getAgentPosition(self.index)
        lines.append("\t\t(at X%dY%d)\n" % (x, y))
        lines.append("\t)\n")

        lines.append("\t(:goal\n")
        # lines.append("\t\t(and\n")
        goal = self.createPDDLGoal(gameState)
        lines.append(goal)
        lines.append("\t)\n")
        lines.append(")\n")

        file.writelines(lines)
        file.close()

    def runPlanner(self):
        cd = os.path.dirname(os.path.abspath(__file__))
        # os.system("%s/%s/ff -o %s/ghostDomain.pddl -f %s/ghostProblem%d.pddl > %s/solution%d.txt"
        #    % (cd, bin_path, cd, cd, self.index, cd, self.index))

        domainFile = 'ghostDomain.pddl'
        problemFile = "%s\ghostProblem%d.pddl" % (cd, self.index)

        data = {
            'domain': open(domainFile).read(),
            'problem': open(problemFile).read()
        }

        # print "Running planner for", domainFile, "and", problemFile

        request = urllib2.Request('http://solver.planning.domains/solve')
        request.add_header('Content-Type', 'application/json')
        response = json.loads(urllib2.urlopen(request, json.dumps(data)).read())

        return response

    def parseSolution(self, response):
        # currentDirectory = os.path.dirname(os.path.abspath(__file__))
        # print currentDirectory
        # file = open("%s/solution%d.txt" % (currentDirectory, self.index), "r")
        # lines = file.readlines()
        # file.close()

        result = response[u'result']

        if u'plan' in result:
            plan = result[u'plan']
            first = plan[0]
            name = first[u'name']
            # print name

            nameSplit = name.split(" ")
            moveToString = nameSplit[2]
            whereIsY = moveToString.find("y")
            whereIsEnd = moveToString.find(")")

            xValue = moveToString[1:whereIsY]
            yValue = moveToString[whereIsY+1:whereIsEnd]

            # print xValue, yValue

            return (int(xValue), int(yValue))

        # pointer = name.find("x")
        return self.getCurrentObservation().getAgentPosition(self.index)

        """
        for line in lines:
            firstAction = line.find("0: ") # First action in solution file
            if firstAction != -1:
                command = line[firstAction:]
                commandSplit = command.split(" ")[3].split(" ")

                x = int(commandSplit[1])
                y = int(commandSplit[2])

                return (x, y)

            # Empty plan, so use STOP action, and return current Position
            if line.find("ff: goal can be simplified to TRUE. The empty plan solves it") != -1:
                return self.getCurrentObservation().getAgentPosition(self.index)

        """


    # A base class for reflex agents that choose score-maximizing actions
    def chooseAction(self, gameState):

        # RUN PLANNER
        self.generatePDDLProblem(gameState)
        response = self.runPlanner()
        (newX, newY) = self.parseSolution(response)

        (x, y) = gameState.getAgentPosition(self.index)
        if newX == x and newY == y:
            return "Stop"
        elif newX == x and newY == y + 1:
            return "North"
        elif newX == x and newY == y - 1:
            return "South"
        elif newX == x + 1 and newY == y:
            return "East"
        elif newX == x - 1 and newY == y:
            return "West"
        else:
            print "ERROR!!!!"

class ClassicalPacmanAgent(PDDLAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        (xSize, ySize) = self.getLayoutDimensions(gameState)
        self.borderLine = 1.0 * xSize / 2
        self.centralLocation = self.findCentralLocation(gameState, xSize, ySize)

        allPositions = gameState.getWalls().asList(False)

        self.objectsString = self.createPDDLObjects(allPositions)
        self.predicatesString = self.createPDDLPredicates(allPositions)

    # GENERATES THE PDDL OBJECTS STRING WHICH IS A LIST OF POSITIONS
    # GENERATES THE PDDL PREDICATES STRING WHICH IS A LIST OF CONNECTED POSITIONS

    def createPDDLObjects(self, positions):
        objects = list()
        objects.append("\t(:objects")

        for position in positions:
            (x, y) = position
            objects.append(" X%dY%d" % (x, y))

        objects.append(" - position)\n")

        return "".join(objects)

    def createPDDLPredicates(self, positions):
        predicates = list()
        predicates.append("\t(:init\n")

        for position in positions:
            (x, y) = position

            if (x - 1, y) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x - 1, y))
            if (x + 1, y) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x + 1, y))
            if (x, y - 1) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x, y - 1))
            if (x, y + 1) in positions:
                predicates.append("\t\t(connected X%dY%d X%dY%d)\n" % (x, y, x, y + 1))

        return "".join(predicates)

    # CURRENT GOAL IS THE NEAREST PACMAN
    def createPDDLGoal(self, gameState):
        goals = list()

        currentPosition = gameState.getAgentPosition(self.index)
        redPositions = self.getPositionsForRed(gameState.getWalls().asList(False))

        # GET THE INDICES OF THE ENEMY AGENTS
        enemies = gameState.getRedTeamIndices()
        for enemy in enemies:
            position = gameState.getAgentPosition(enemy)
            if position != None and position[0] < self.borderLine + 1:
                (x, y) = position
                goals.append("\t\t\t(not (visited X%dY%d))\n" % (x, y))
                if (x - 1, y) in redPositions and not (x - 1, y) == currentPosition:
                    goals.append("\t\t\t(not (visited X%dY%d))\n" % (x - 1, y))
                if (x + 1, y) in redPositions and not (x + 1, y) == currentPosition:
                    goals.append("\t\t\t(not (visited X%dY%d))\n" % (x + 1, y))
                if (x, y - 1) in redPositions and not (x, y - 1) == currentPosition:
                    goals.append("\t\t\t(not (visited X%dY%d))\n" % (x, y - 1))
                if (x, y + 1) in redPositions and not (x, y + 1) == currentPosition:
                    goals.append("\t\t\t(not (visited X%dY%d))\n" % (x, y + 1))

        redFood = gameState.getRedFood().asList()

        """
        closestEnemyPosition = None
        closestEnemyDistance = 9999

        for enemy in enemies:
            position = gameState.getAgentPosition(enemy)
            if position != None and position[0] > self.borderLine:
                distance = self.distancer.getDistance(currentPosition, position)
                if distance < closestEnemyDistance:
                    closestEnemyPosition = position
                    closestEnemyDistance = distance

        if closestEnemyPosition != None:
            (goalX, goalY) = closestEnemyPosition
            return "\t\t(at X%dY%d)\n" % (goalX, goalY)
        """

        """
        closestFood = None
        closestFoodDistance = 9999

        for food in redFood:
            distance = self.distancer.getDistance(currentPosition, food)
            if distance < closestFoodDistance:
                closestFood = food
                closestFoodDistance = distance

        if closestFood != None:
            (x, y) = closestFood
            goals.append("\t\t\t(visited X%dY%d)\n" % (x, y))
        """

        if gameState.getAgentState(self.index).numCarrying == 0:
            nextFood = redFood[-1]
            (x, y) = nextFood
            goals.append("\t\t\t(visited X%dY%d)\n" % (x, y))

        goals.append("\t\t\t(carryingFood)\n")

        allPositions = gameState.getWalls().asList(False)

        homePositions = [position for position in allPositions if position[0] == int(self.borderLine + 1)]
        goals.append("\t\t\t(or\n")
        for position in homePositions:
            (x, y) = position
            goals.append("\t\t\t\t(at X%dY%d)\n" % (x, y))
        goals.append("\t\t\t)\n")

        return "".join(goals)


        # FILL THE CODE TO GENERATE PDDL GOAL
        # (goalX, goalY) = goalPosition
        (midX, midY) = self.centralLocation
        return "\t\t(at X%dY%d)\n" % (midX, midY)

    def generatePDDLProblem(self, gameState):
        # CONVERT A SCENARIO TO A STRIPS PROBLEM
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        # file = open("%s/pacmanProblem%d.pddl" % (currentDirectory, self.index), "w")  # Windows
        file = open("%s\pacmanProblem%d.pddl" % (currentDirectory, self.index), "w")    # Mac or Unix

        lines = list()
        lines.append("(define (problem pacman%d)\n" % self.index)
        lines.append("\t(:domain pacman)\n")
        lines.append(self.objectsString)
        # lines.append(")\n")

        lines.append(self.predicatesString)

        (x, y) = gameState.getAgentPosition(self.index)
        lines.append("\t\t(at X%dY%d)\n" % (x, y))
        lines.append("\t\t(visited X%dY%d)\n" % (x, y))

        redFood = gameState.getRedFood().asList()
        for food in redFood:
            (x, y) = food
            lines.append("\t\t(hasFood X%dY%d)\n" % (x, y))

        if gameState.getAgentState(self.index).numCarrying > 0:
            lines.append("\t\t(carryingFood)\n")


        lines.append("\t)\n")

        lines.append("\t(:goal\n")
        lines.append("\t\t(and\n")
        # lines.append("\t\t(and\n")
        goal = self.createPDDLGoal(gameState)
        lines.append(goal)
        lines.append("\t\t)\n")
        lines.append("\t)\n")
        lines.append(")\n")

        file.writelines(lines)
        file.close()

    def runPlanner(self):
        cd = os.path.dirname(os.path.abspath(__file__))
        # os.system("%s/%s/ff -o %s/ghostDomain.pddl -f %s/ghostProblem%d.pddl > %s/solution%d.txt"
        #    % (cd, bin_path, cd, cd, self.index, cd, self.index))

        domainFile = 'pacmanDomain.pddl'
        problemFile = "%s\pacmanProblem%d.pddl" % (cd, self.index)

        data = {
            'domain': open(domainFile).read(),
            'problem': open(problemFile).read()
        }

        # print "Running planner for", domainFile, "and", problemFile

        request = urllib2.Request('http://solver.planning.domains/solve')
        request.add_header('Content-Type', 'application/json')
        response = json.loads(urllib2.urlopen(request, json.dumps(data)).read())

        return response

    def parseSolution(self, response):
        # currentDirectory = os.path.dirname(os.path.abspath(__file__))
        # print currentDirectory
        # file = open("%s/solution%d.txt" % (currentDirectory, self.index), "r")
        # lines = file.readlines()
        # file.close()

        result = response[u'result']

        if u'plan' in result:
            plan = result[u'plan']
            first = plan[0]
            print first

            name = first
            # name = first[u'name'] => SEEMED TO WORK FOR GHOST FORMAT
            # print name

            nameSplit = name.split(" ")
            moveToString = nameSplit[2]
            whereIsY = moveToString.find("y")
            whereIsEnd = moveToString.find(")")

            xValue = moveToString[1:whereIsY]
            yValue = moveToString[whereIsY+1:whereIsEnd]

            # print xValue, yValue

            return (int(xValue), int(yValue))

        # pointer = name.find("x")
        return self.getCurrentObservation().getAgentPosition(self.index)

        """
        for line in lines:
            firstAction = line.find("0: ") # First action in solution file
            if firstAction != -1:
                command = line[firstAction:]
                commandSplit = command.split(" ")[3].split(" ")

                x = int(commandSplit[1])
                y = int(commandSplit[2])

                return (x, y)

            # Empty plan, so use STOP action, and return current Position
            if line.find("ff: goal can be simplified to TRUE. The empty plan solves it") != -1:
                return self.getCurrentObservation().getAgentPosition(self.index)

        """


    # A base class for reflex agents that choose score-maximizing actions
    def chooseAction(self, gameState):

        # RUN PLANNER
        self.generatePDDLProblem(gameState)
        response = self.runPlanner()
        (newX, newY) = self.parseSolution(response)

        (x, y) = gameState.getAgentPosition(self.index)
        if newX == x and newY == y:
            return "Stop"
        elif newX == x and newY == y + 1:
            return "North"
        elif newX == x and newY == y - 1:
            return "South"
        elif newX == x + 1 and newY == y:
            return "East"
        elif newX == x - 1 and newY == y:
            return "West"
        else:
            print "ERROR!!!!"
