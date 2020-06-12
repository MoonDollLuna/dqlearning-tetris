# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

from agents.old.dql_agent_old import DQLAgentOld

# General imports
from collections import deque
import numpy as np
import random
import csv
from os import mkdir
from os.path import exists, join

# Keras related imports
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import glorot_uniform


class WeightedAgentOld(DQLAgentOld):
    """
    This class is a variation of the standard DQL Agent used by the original approach (action = player input), but
    with specific weights when acting randomly (in the Act method)

    The new weights are as following:
    *   25% chance of moving left
    *   25% chance of moving right
    *   40% chance of rotating the piece
    *   10% chance of dropping the piece

    The idea is to artificially force the agent to perform more rotations and less instant drops (the main problem of
    the original agent), to try to have it experience a wider pool of states.
    """

    def act(self, state):
        """
        For the current state, return the optimal action to take or a random action randomly

        Note that, as specified above, the random action chances are weighted instead of uniform, following
        these weights:

        *   25% chance of moving left
        *   25% chance of moving right
        *   40% chance of rotating the piece
        *   10% chance of dropping the piece

        :param state: The current state provided by the game
        :return: The action taken (as a string) and, if applicable, the set of Q-Values for each action
        """

        # Count the action
        self.actions_performed += 1

        # Prepare the state for the neural network
        state = np.expand_dims(state, axis=0)
        # Predict the q-values for the state (will be needed anyways to keep track of the values)
        q_values = self.q_network.predict(state)

        # Generate a random number
        random_chance = np.random.rand()

        # Check if the value is smaller (random action) or greater (optimal action) than epsilon
        if random_chance < self.epsilon:
            # Take a random action from the actions dictionary

            # Weighting is applied, to force more rotations and less drops
            # Reminder that the dictionary structure of actions is as follows (in this order):
            #       * 0: right
            #       * 1: left
            #       * 2: rotate
            #       * 3: hard_drop
            action = np.random.choice(list(self.actions.keys()), p=[0.25, 0.25, 0.4, 0.1])

            # Add the appropriate value to the value counter and return the action
            # (no Q-Values will be returned in this case)
            self.q_values += q_values[0][action]
            return self.actions[action], None

        else:
            # Choose the best action and add the value to the value counter
            action = np.argmax(q_values[0])
            self.q_values += q_values[0][action]
            return self.actions[action], q_values

