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

    def __init__(self, learning_rate, alpha, gamma, epsilon, batch_size, seed):
        """
        Constructor of the class. Creates an agent from the specified information

        :param learning_rate: Learning rate for the model
        :param alpha: Initial alpha value (weight given to the new value)
        :param gamma: Initial gamma value (discount factor, importance given to future rewards)
        :param epsilon: Initial epsilon value (chance for a random action in exploration-exploitation)
        :param batch_size: How many actions will be sampled at once
        :param seed: Seed to be used for all random choices. Optional.
        """

        # Create a dictionary to link every output to the agent to an actual action
        self.actions = {
            0: 'right',
            1: 'left',
            2: 'rotate',
            3: 'drop'
        }

        # Set the values of the algorithm variables
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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
        # Activation function is softmax (converts values into a probability, adding all outputs returns 1)
        output_layer = Dense(len(self.actions),
                             activation="softmax",
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

    def act(self, state):
        """
        For the current state, return the optimal action to take or a random action randomly
        :param state: The current state provided by the game
        :return: The action taken (as a string)
        """

        # Generate a random number
        random_chance = np.random.rand()

        # Check if the value is smaller (random action) or greater (optimal action) than epsilon
        if random_chance < self.epsilon:
            # Take a random action from the actions dictionary
            return np.random.choice(self.actions.values())
        else:
            # Take the optimal action
            q_values = self.q_network.predict(state)
            return self.actions[np.argmax(q_values[0])]

    def insert_experience(self, state, action, reward, next_state, terminated):
        """
        Creates an experience and stores it into the experience replay of the agent

        :param state: Initial state
        :param action: Action taken in the initial state
        :param reward: Reward of taking the action in the initial state
        :param next_state: State reached from taking the action from the initial state
        :param terminated: Whether the initial state is a final state or not
        """

        # Store everything as a tuple
        self.experience_replay.append((state, action, reward, next_state, terminated))

        # Train the network
        self._learn_from_replay()

    def finish_epoch(self):
        """
        Finishes the current epoch, updating all necessary values
        """

        # Update the policy network to the current Q network weights
        self._update_target_network()


