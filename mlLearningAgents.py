# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent, Directions
from pacman_utils import util
from pacman_utils.util import flipCoin
import numpy as np

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        
        # Extracting necessary information from the game state
        
        # Pacman's current position
        self.pacmanPos = state.getPacmanPosition() 
        # Positions of all ghosts
        self.ghostPositions = state.getGhostPositions() 
        # Grid containing food positions
        self.food = state.getFood()  
        # Positions of capsules
        self.capsules = state.getCapsules()
        # Grid representing walls
        self.walls = state.getWalls()
        # Whether the game is won
        self.isWin = state.isWin() 
        # Whether the game is lost
        self.isLose = state.isLose()  
        
        
    def __hash__(self):
        # Hash function for state representation
        # Hashing Pacman and ghost positions
        return hash((self.pacmanPos, tuple(self.ghostPositions))) 
        
        
class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state 
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts) 
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        
        # Initialise an empty Q-value table
        self.qValues = {}   
        # Initialise an empty table to track the number of visits to each state-action pair
        self.stateActionCounts = {} 
        
        # last state
        self.lastState = []
        # last action
        self.lastAction = []
        
    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
            
     
        # Difference between endState score and startState score: Base reward
        reward = endState.getScore() - startState.getScore()
    
        # If Pacman eats food: Additional reward
        if endState.getNumFood() < startState.getNumFood():
            reward += 10

        # If Pacman eats a capsule: Additional reward
        if len(endState.getCapsules()) < len(startState.getCapsules()):
            reward += 200  
    
        # If Pacman dies: Penalty
        if endState.isLose():
            reward -= 500 

        # If Pacman wins the game: Additional reward
        if endState.isWin():
            reward += 500 
        
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"


        key = (state.__hash__(), action)
        
        # Check if this (state, action) pair is in the qValues dictionary
        if key in self.qValues:
            # Return the stored Q-value for the pair
            return self.qValues[key] 
        else:
            self.qValues[key] = 0
            # If the pair is not recorded, initialise with default Q-value 0
            return self.qValues[key]
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"

        # Retrieve all possible actions in the given state
        legalActions = state.getLegalActions()
        
        # If there are no legal actions in the given state (e.g., terminal state), return 0.0 immediately
        if not legalActions:
            return 0.0
        
        # Utilize list to gather Q-values for all actions, then return the maximum value
        lst = []
        for action in legalActions:
            lst.append(self.getQValue(state, action))
            
        maxqvalue = max(lst)
        
        return maxqvalue
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        
        # Calculate the current Q-value for the (state, action) pair
        currentQValue = self.getQValue(state, action)
        # Calculate the maximum Q-value for the next state
        nextMaxQValue = self.maxQValue(nextState)
        # Q-learning update rule
        # Q(s, a) <- (1 - learning rate) * Q(s, a) + learning rate * (reward + discount factor * max_a' Q(s', a')) 
        newQValue = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.gamma * nextMaxQValue)
          
        key = (state.__hash__(), action)
        self.qValues[key] = newQValue
        self.updateCount(state,action)
        
        # Print out the intermediate values for debugging and analysis
        #print("currentQValue is",currentQValue)
        #print("nextMaxQValue is", nextMaxQValue)
        #print("newQValue is",newQValue )
        
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"

        key = (state.__hash__(), action)
        
        # Update the visit count
        if key in self.stateActionCounts:
            self.stateActionCounts[key] += 1
        else:
            self.stateActionCounts[key] = 1


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        
        key = (state.__hash__(), action)
        
        # Return the visit count for the corresponding (state, action) pair 
        # If the pair is not recorded, return 0
        return self.stateActionCounts.get(key, 0)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        

        if counts <= self.maxAttempts:
            # Use the sqrt of the inverse frequency, encouraging exploration
            # of less frequently explored actions. Adding 1 to counts to avoid division by zero
            explore_bonus = (self.maxAttempts / (counts + 1))**0.1  
            return utility + explore_bonus
        else:
            # No exploration bonus if the counts exceed maxAttempts
            return utility  


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:

        # Retrieve all possible actions in the given state
        legalActions = state.getLegalPacmanActions()
        
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        
        if not legalActions:
            return Directions.STOP
        
        
        # Decide action based on exploration or exploitation
        if flipCoin(self.epsilon):
            # Exploration: Choose randomly from legal moves
            bestAction = random.choice(legalActions)
        else:
            # During the first half of the training, avoid stopping or reversing when not near a ghost
            legalActions = state.getLegalPacmanActions()
            if self.getEpisodesSoFar()*1.0/self.getNumTraining() < 0.5:
                if len(self.lastAction) > 0:
                    if Directions.STOP in legalActions:
                        legalActions.remove(Directions.STOP)
                    last_action = self.lastAction[-1]
                    distance0 = state.getPacmanPosition()[0] - state.getGhostPosition(1)[0]
                    distance1 = state.getPacmanPosition()[1] - state.getGhostPosition(1)[1]
                    if np.sqrt(distance0**2 + distance1**2) > 2:
                        if (Directions.REVERSE[last_action] in legalActions) and len(legalActions) > 1:
                            legalActions.remove(Directions.REVERSE[last_action])
                            
                            
            # Exploitation: Choose the best action based on exploration Values
            utilities = []
            
            for action in legalActions:
                qValue = self.getQValue(state, action)
                #print("Q-value is",qValue)
                count = self.getCount(state, action)
                explorationValue = self.explorationFn(qValue, count)
                utilities.append((explorationValue, action))
                
            #print(utilities)
            
            bestexplorationValue, bestAction = max(utilities)    
            
            
        if len(self.lastState) > 0:
            # Compute the reward for the transition from the last state to the current state
            reward = self.computeReward(self.lastState[-1], state)
            # Retrieve the last state and action
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            #print("reward is", reward) 
            # Update the Q-value based on the reward received
            self.learn(last_state,last_action,reward,state)
        
        self.updateCount(state,bestAction)
                
        # update attributes
        self.lastState.append(state)
        self.lastAction.append(bestAction)
        #print("lastState is", self.lastState)
        #print("lastAction is", self.lastAction)
        #print("Q-table is", self.qValues)
        
        return bestAction
        
        
        
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        
        # Learn the last state and action of the game
        if len(self.lastState) > 0:
            reward = self.computeReward(self.lastState[-1], state)
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            #print("reward is", reward) 
            self.learn(last_state,last_action,reward,state)
            
        # reset attributes
        self.lastState = []
        self.lastAction = []
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
        
        