# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

# General imports
from collections import deque
import numpy as np
import random

# Keras related imports
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import glorot_uniform


class DQLAgent:
    """
    This class represents a basic Deep Q-Learning Agent, including all relevant and necessary methods.

    This agent implements:
    - A prediction network
    - A target network (fixed Q-Targets)
    - Experience replay
    - Epsilon-greedy policy for selecting the action (exploration-exploitation)
    """

    def __init__(self, learning_rate, gamma, epsilon, epsilon_decay, batch_size, seed):
        """
        Constructor of the class. Creates an agent from the specified information

        :param learning_rate: Learning rate for the model
        :param gamma: Initial gamma value (discount factor, importance given to future rewards)
        :param epsilon: Initial epsilon value (chance for a random action in exploration-exploitation)
        :param epsilon_decay: Decay value for epsilon (how much epsilon decreases every epoch, linearly)
        :param batch_size: How many actions will be sampled at once
        :param seed: Seed to be used for all random choices. Optional.
        """

        # Create a dictionary to link every output to the agent to an actual action
        self.actions = {
            0: 'right',
            1: 'left',
            2: 'rotate',
            3: 'hard_drop'
        }

        # Create an inverse dictionary, to be able to look up numeric values by name
        self.inverse_actions = {
            'right': 0,
            'left': 1,
            'rotate': 2,
            'hard_drop': 3
        }

        # Store variables related to DQL
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon

        # Creates the experience replay container
        # A deque is used to have a queue (oldest experiences in the experience replay go out first)
        # We specify a max-length to the deque, to ensure that the experience replay doesn't grow indefinitely
        # and that the experiences cycle

        self.experience_replay = deque(maxlen=2000)

        # Store the seed and set it, if it has been provided.
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Create the actual neural network
        self.q_network = self._construct_neural_network(learning_rate)
        # Copy the network for the target network
        # We will create a network with the same structure and then transfer the weights
        # from the Q-Network to the Policy Network (much easier than actually cloning the network)
        self.target_network = self._construct_neural_network(learning_rate)
        self._update_target_network()

        # Set the batch size
        self.batch_size = batch_size

        # Internal variables #

        # Count the actual epoch
        self.current_epoch = 1

        # Keep a track of the count of actions performed this epoch
        self.actions_performed = 0



    # Internal methods

    def _construct_neural_network(self, learning_rate):
        """
        Generates the neural network to be used as the Q-Network and Policy Network.

        NOTE: Even when using a seed, due to the parallelization done by Keras,
        we are not always guaranteed to obtain the same results

        :param learning_rate: Learning rate for the model
        :return: a Keras model, already compiled and with the appropriate weights initialized
        """

        # The neural network in this case is a simple multilayer perceptron
        # We don't think that convolution will be needed, since the input will be small
        # (20x10 with 3 possible values for each position)
        # No dropout will be used (we are interested in the correlations)

        flatten_layer = Flatten(data_format="channels_last",
                                input_shape=(20, 10))

        # Two dense hidden layers will be used
        # Activation function is ReLU (standard activation for deep networks)
        # Weights are initialized a uniform Glorot and Bengio initializer (standard for deep networks)
        hidden_layer_1 = Dense(64,
                               activation="relu",
                               kernel_initializer=glorot_uniform(seed=self.seed))
        hidden_layer_2 = Dense(64,
                               activation="relu",
                               kernel_initializer=glorot_uniform(seed=self.seed))

        # Output layer has 4 neurons (one for each possible action)
        # Activation function is linear (instead of the usual softmax): we want Q-Values, not probabilities
        output_layer = Dense(len(self.actions),
                             activation="linear",
                             kernel_initializer=glorot_uniform(seed=self.seed))

        # Create the sequential model
        nn_layers = [flatten_layer, hidden_layer_1, hidden_layer_2, output_layer]
        nn_model = Sequential(nn_layers)

        # Compile the model
        # Adam is used as an optimizer (standard for stochastic optimization)
        # Loss will be mean squared error (error used to update weights in Deep Q-Learning)

        nn_model.compile(optimizer=Adam(lr=learning_rate),
                         loss="mse")

        return nn_model

    def _update_target_network(self):
        """
        Transfers the weights from the Q-Network to the target network
        """

        self.target_network.set_weights(self.q_network.get_weights())

    def _learn_from_replay(self):
        """
        Train the Q-Network using experiences from the experience replay

        The network is trained after every action
        """
        # TODO TRABAJA EN BATCH
        # Take a batch from the experience replay
        if len(self.experience_replay) < self.batch_size:
            size = len(self.experience_replay)
        else:
            size = self.batch_size

        batch = random.sample(self.experience_replay, size)

        # Train the network using the sampled experiences
        for state, action, reward, next_state, terminated in batch:

            # Obtain the expected actions
            train = self.q_network.predict(state)

            #TODO PON AQUI PRINTS, QUE QUIERO VER COMO FUNCHA ESTO

            # Check if the experience was a final one
            if terminated:
                # Final state - The q value is only considered as a reward
                train[0][action] = reward
            else:
                # Not a final state - We need to consider the max Q value in the next state
                target_actions = self.target_network.predict(next_state)
                train[0][action] = reward + self.gamma * np.amax(target_actions)

            # Train the network (verbose 0 to ensure that no messages are shown)
            self.q_network.fit(state, train, epochs=1, verbose=0)

    @staticmethod
    def _add_extra_dimension(state):
        """
        Adds an extra dimension to a state, transforming it into Keras format

        :param state: Original state to transform
        :return: Transformed state (with an extra dimension)
        """

        return np.expand_dims(state, axis=0)

    # Public methods

    def act(self, state):
        """
        For the current state, return the optimal action to take or a random action randomly
        :param state: The current state provided by the game
        :return: The action taken (as a string) and, if applicable, the set of chances for each action
        """

        # Count the action
        self.actions_performed += 1

        # Generate a random number
        random_chance = np.random.rand()

        # Prepare the state to pass through the neural network
        state = self._add_extra_dimension(state)

        # Check if the value is smaller (random action) or greater (optimal action) than epsilon
        if random_chance < self.epsilon:
            # Take a random action from the actions dictionary
            # The strings are directly sampled in this case
            return np.random.choice(list(self.actions.values())), None
        else:
            # Take the optimal action
            q_values = self.q_network.predict(state)
            return self.actions[np.argmax(q_values[0])], q_values

    def insert_experience(self, state, action, reward, next_state, terminated):
        """
        Creates an experience and stores it into the experience replay of the agent

        If enough experiences have been inserted, train the Q Network

        :param state: Initial state
        :param action: Action taken in the initial state
        :param reward: Reward of taking the action in the initial state
        :param next_state: State reached from taking the action from the initial state
        :param terminated: Whether the initial state is a final state or not
        """

        # Prepare the states with Keras format
        state = self._add_extra_dimension(state)
        next_state = self._add_extra_dimension(next_state)

        # Convert the action back into its numeric position
        action = self.inverse_actions[action]

        # Store everything as a tuple
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def finish_epoch(self, lines, score):
        """
        Finishes the current epoch, updating all necessary values
        """

        # Train the network
        self._learn_from_replay()

        # Update the policy network to the current Q network weights
        self._update_target_network()

        # Update the epsilon with the epsilon decay (and check that it doesn't go below 0)
        self.epsilon = self.epsilon - self.epsilon_decay * self.current_epoch
        if self.epsilon < 0.0:
            self.epsilon = 0.0

        # Print the relevant info on the screen
        print("EPOCH " + str(self.current_epoch) + " FINISHED (Lines: " + str(lines) + "/Score: " + str(score) + ")")

        # Store the info for the current epoch
        # TODO

        # Update the epoch
        self.current_epoch += 1

