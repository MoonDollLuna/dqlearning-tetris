# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

from agents.new.dql_agent_new import DQLAgentNew

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


class PrioritizedAgentNew(DQLAgentNew):
    """
    This class is a variation of the standard DQL Agent used by the new approach, that has been modified
    to use Prioritized Experience Replay (giving priority to experiences with a higher error)

    The Replay Memory will have a new structure in its queue, each experience being stored as:
        (error, experience)

    A sorted version of this queue will be created after every insertion, in order to use rank-based PER
    Since the sorting is done on a separate list, the queue structure is conserved
    This means that the queue will work as expected (once it is full, the oldest experiences will be removed)

    Experiences are initially inserted with an infinite error (to give them higher priority)

    Error is only updated when the experiences are sampled

    PER is supposed to improve the performance of DQL, by allowing the most relevant experiences to be
    replayed more frequently
    """

    def __init__(self, learning_rate, gamma, epsilon, epsilon_decay, minimum_epsilon,
                 batch_size, total_epochs, experience_replay_size, seed, rewards_method):
        """
        Constructor of the class. Creates an agent from the specified information, overriding the appropiate info

        To be more precise, the PER elements (alpha and beta) are added

        :param learning_rate: Learning rate for the model
        :param gamma: Initial gamma value (discount factor, importance given to future rewards)
        :param epsilon: Initial epsilon value (chance for a random action in exploration-exploitation)
        :param epsilon_decay: Decay value for epsilon (how much epsilon decreases every epoch, linearly)
        :param minimum_epsilon: Minimum value for epsilon (cannot go below this value)
        :param batch_size: How many actions will be sampled at once
        :param total_epochs: Total amount of epochs that will be performed
        :param experience_replay_size: The maximum size of the experience replay
        :param seed: Seed to be used for all random choices. If None, a random seed will be used
        :param rewards_method: Method used to compute the reward. Only used to differentiate when storing results
        """

        # Call the super constructor
        DQLAgentNew.__init__(self, learning_rate, gamma, epsilon, epsilon_decay, minimum_epsilon,
                             batch_size, total_epochs, experience_replay_size, seed, rewards_method)

        # Set alpha and beta
        self.alpha = 0.5
        self.beta = 0.4

        # Keep track of the insertion order (used to ensure that the sort is stable)
        self.insertion = 0

        # Keep the last sorted list to access it outside
        self.sorted_queue = None

        # Compute the beta increment to ensure that beta reaches 1 after 75% of epochs
        self.beta_increment = (1 - 0.4) / (total_epochs * 0.75)

    # Internal methods
    def insert_experience(self, state, reward, next_state, terminated):
        """
        Creates an experience and stores it into the replay memory of the agent

        The agent will be trained after inserting the experience, using a mini batch

        The inserted experiences will now have a different structure:
        (error, insertion, experience)

        Initial error inserted is infinite

        After inserting the experience, a sorted version of the queue will be generated
        This will later be used to perform rank-based PER

        :param state: Initial state
        :param reward: Reward of taking the action in the initial state
        :param next_state: State reached from taking the action from the initial state
        :param terminated: Whether the initial state is a final state or not
        """

        # Create the experience
        experience = (state, reward, next_state, terminated)

        # Store the experience as a tuple
        self.experience_replay.append((float('inf'), self.insertion, experience))

        # Create the ordered queue
        self.sorted_queue = sorted(self.experience_replay, key=lambda order: (order[0], order[1]), reverse=True)

        # Train the network
        self._learn_from_replay()

    def _learn_from_replay(self):
        """
        Train the Q-Network using experiences from the experience replay

        In this case, the sorted experience replay will be used for PER

        The network is trained after every action
        """

        # Compute the rank of every element
        # Since they are ordered, this can be done algorithmically
        ranks = [(1/x) for x in range(1, len(self.sorted_queue) + 1)]

        # Compute the addition of all ranks to the power of alpha
        addition = 0.0
        for r in ranks:
            addition += r ** self.alpha

        # Compute the probability of every element (rank-based)
        probabilities = [((rank ** self.alpha)/ addition) for rank in ranks]

        # Take a batch from the experience replay
        # In this case, we're sampling indexes from the ordered list
        if len(self.sorted_queue) < self.batch_size:
            size = len(self.sorted_queue)
        else:
            size = self.batch_size

        # Sample the sorted experience replay using the weights
        # Sampling without replacement is used
        print(self.sorted_queue)

        batch = np.random.choice(self.sorted_queue, size, False, probabilities)

        # Generate a list with all states and next states (in batch order)
        states = [x[0] for x in batch]
        next_states = [x[2] for x in batch]

        # Create a list to store the predictions
        states_predictions = []

        # Batch predict the Q-Values for the next states (using the Target Network)
        next_states_predictions = self.target_network.predict(np.array(next_states), batch_size=self.batch_size, verbose=0)

        # PER: Obtain the current predictions
        current_predictions = self.q_network.predict(np.array(states), batch_size=self.batch_size, verbose=0)
        # PER: Create a list to store the updated errors
        errors = []

        # Compute the updated Q values
        for i in range(len(states)):
            # Grab the missing values from the batch
            # Batch structure = (state, reward, next_state, terminated)
            reward = batch[i][1]
            terminated = batch[i][3]

            # Check if the experience was a final one
            if terminated:
                # Final state - The Q-value is only considered as a reward
                new_value = reward
            else:
                # Not a final state - We need to consider the max Q value in the next state
                new_value = reward + self.gamma * next_states_predictions[i]

            # Compute the error with the weight
            weight = ((1 / len(self.experience_replay)) * (1/probabilities[i])) ** self.beta
            errors.append((current_predictions[i] - new_value) ** 2)
            weighted_change = weight * new_value

            # Append the weighted change
            states_predictions.append(weighted_change)


        # Batch train the network
        self.q_network.fit(np.array(states), np.array(states_predictions), batch_size=self.batch_size, epochs=1,
                           verbose=0)

        # After training the network, update the experiences already in the queue
        for experience in range(len(batch)):
            # Find the experience in the queue
            exp_id = self.experience_replay.index(batch[experience])

            # Update the error
            self.experience_replay[exp_id][0] = errors[experience]


