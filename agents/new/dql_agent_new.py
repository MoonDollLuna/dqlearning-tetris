# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

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


class DQLAgentNew:
    """
    This class represents a basic Deep Q-Learning Agent, including all relevant and necessary methods.

    This agent implements:
    - A prediction network
    - A target network (fixed Q-Targets)
    - Experience replay
    - Epsilon-greedy policy for selecting the action (exploration-exploitation)

    In contrast with the "old" approach, this agent considers an action as the final position of the piece.
    The agent will receive all actions (all posible rotations and positions for the piece) and will score every action.
    The action with the best score will be returned as the action chosen
    """

    def __init__(self, learning_rate, gamma, epsilon, epsilon_decay, minimum_epsilon,
                 batch_size, total_epochs, experience_replay_size, seed, rewards_method):
        """
        Constructor of the class. Creates an agent from the specified information

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

        # Store variables related to DQL
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon

        # Creates the experience replay container
        # A deque is used to have a queue (oldest experiences in the experience replay go out first)
        # We specify a max-length to the deque, to ensure that the experience replay doesn't grow indefinitely
        # and that the experiences cycle

        self.experience_replay = deque(maxlen=experience_replay_size)

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

        # Store the total epochs to be performed
        self.total_epochs = total_epochs

        # Store the rewards method being used
        self.rewards_method = rewards_method

        # Internal variables #

        # Remember the initial epsilon
        self.initial_epsilon = self.epsilon

        # Count the actual epoch
        self.current_epoch = 1

        # Keep a track of the count of actions (final positions) (player inputs) performed this epoch
        # The number of steps will actually be tracked by the game itself
        self.actions_performed = 0

        # Mark this agent as a NEW agent (using the new approach, where Action = Final piece position)
        self.agent_type = "new"

        # Compute the class and file name (used to store results)
        self.class_name = self.__class__.__name__
        self.folder_name = "g" + str(self.gamma) + "eps" + str(self.initial_epsilon) + "seed" + str(self.seed) + "epo" + str(self.total_epochs) + "rew" + self.rewards_method

        # Keep track of the steps taken (displacements to the piece)
        self.displacements = 0

    # Internal methods

    def _return_version(self):
        """
        Returns the version. Used to distinguish between agents of the old and new implementations.
        :return: the type of the agent
        """

        return self.agent_type

    def _construct_neural_network(self, learning_rate):
        """
        Generates the neural network to be used as the Q-Network and Policy Network.

        NOTE: Even when using a seed, due to the parallelization done by Keras,
        we are not always guaranteed to obtain the same results

        The structure of the network is as follows:
        INPUT: A 20x10 Flatten layer (converts the matrix into a single array)
        HIDDEN LAYERS: 2 64-neuron ReLU Dense layers (standard for Deep Q-Learning)
        OUTPUT: A 1-neuron Linear Dense layer (only one output, since the network only needs to output the state score)

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

        # Output layer has 1 neuron (the score of the current state)
        # Activation function is linear
        output_layer = Dense(1,
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
        # Take a batch from the experience replay
        if len(self.experience_replay) < self.batch_size:
            size = len(self.experience_replay)
        else:
            size = self.batch_size

        batch = random.sample(self.experience_replay, size)

        # Generate a list with all states and next states (in order)
        states = [x[0] for x in batch]
        next_states = [x[2] for x in batch]

        # Create a list to store the predictions
        states_predictions = []

        # Batch predict the Q-Values for the next states (using the Target Network)
        next_states_predictions = self.target_network.predict(np.array(next_states), batch_size=self.batch_size, verbose=0)

        # Compute the updated Q values
        for i in range(len(states)):
            # Grab the missing values from the batch
            # Batch structure = (state, reward, next_state, terminated)
            reward = batch[i][1]
            terminated = batch[i][3]

            # Check if the experience was a final one
            if terminated:
                # Final state - The Q-value is only considered as a reward
                states_predictions.append(reward)
            else:
                # Not a final state - We need to consider the max Q value in the next state
                states_predictions.append(reward + self.gamma * next_states_predictions[i])

        # Batch train the network
        self.q_network.fit(np.array(states), np.array(states_predictions), batch_size=self.batch_size, epochs=1, verbose=0)

    # Public methods

    def act(self, actions):
        """
        For the current state, return the optimal action to take or a random action randomly

        :param actions: A list of all possible actions, with structure:
                 (x_position, rotation, state)
        :return: action_taken, q-value
                 WHERE
                 action_taken: The action taken, passed as (x_position, rotation, state)
                 q-value: The q-value (score) given to the action
        """

        # Count the action
        self.actions_performed += 1

        # Prepare the states for the network and pass them in batch
        states = [x[2] for x in actions]
        q_values = self.q_network.predict(np.array(states), batch_size=len(actions), verbose=0)

        # Generate a random number
        random_chance = np.random.rand()

        # Check if the value is smaller (random action) or greater (optimal action) than epsilon
        if random_chance < self.epsilon:
            # Directly take a random action from the list of actions
            action = np.random.choice(np.arange(0, len(actions)))
            # Find the appropriate Q-value
            q_value = q_values[action]

            return actions[action], q_value

        else:
            # Find the best action
            best_action_id = np.argmax(q_values)
            # Find the appropriate Q-value
            q_value = q_values[best_action_id]

            return actions[best_action_id], q_value

    def insert_experience(self, state, reward, next_state, terminated):
        """
        Creates an experience and stores it into the replay memory of the agent

        The agent is trained after inserting the experience with a mini batch

        Note that no actions are stored into the replay memory. This is because the action is implicitly stored
        within the next state (since actions are the final positions of the pieces)

        :param state: Initial state
        :param reward: Reward of taking the action in the initial state
        :param next_state: State reached from taking the action from the initial state
        :param terminated: Whether the initial state is a final state or not
        """

        # Store everything as a tuple
        self.experience_replay.append((state, reward, next_state, terminated))

        # Train the network
        self._learn_from_replay()

    def initialize_learning_structure(self):
        """
        Creates the necessary folder structure used to store all results, and creates a new CSV to store the results

        The folder structure is the following:
        results =>
            <AGENT NAME> =>
                g<GAMMA VALUE>eps<EPSILON VALUE>epo<TOTAL EPOCHS> =>
                    <AGENT NAME>_g<GAMMA VALUE>eps<EPSILON VALUE>epo<TOTAL EPOCHS>_data.csv
                    weights =>
                        [stored weights learned during the epochs]
        """

        # If results does not exist, create the folder
        if not exists("results"):
            mkdir("results")

        # If the agent name does not exist within results, create the folder
        if not exists(join("results", self.class_name)):
            mkdir(join("results", self.class_name))

        # If there is not a folder for this specific configuration, create it
        if not exists(join("results", self.class_name, self.folder_name)):
            mkdir(join("results", self.class_name, self.folder_name))

        # If the weights folder does not exist within this specific instance of the agent, create it
        if not exists(join("results", self.class_name, self.folder_name, "weights")):
            mkdir(join("results", self.class_name, self.folder_name, "weights"))

        # Create a new CSV to store the results
        # Since the file is created freshly every time, we are guaranteed that it will exist later
        with open(join("results", self.class_name, self.folder_name, self.class_name + "_" + self.folder_name + "_data.csv"), 'w', newline='') as file:
            # Create the writer
            writer = csv.writer(file)
            # Create the column names
            writer.writerow(["epoch", "score", "lines", "actions"])

    def load_weights(self, weights):
        """
        Loads the pre-trained weights from the given file. If no file is passed, nothing is done

        :param weights: Path to the file containing the weights.
        """

        # Only act if there is an actual file
        if weights is not None:
            # Load the weights into both networks
            self.q_network.load_weights(weights)
            self._update_target_network()
            print("Weights successfully loaded")

    def notify_step(self):
        """
        Increases the counter of steps (displacements). Called by the loop every time a step is taken
        """

        self.displacements += 1

    def finish_epoch(self, lines, score):
        """
        Finishes the current epoch, updating all necessary values
        """

        # Train the network
        self._learn_from_replay()

        # Update the policy network to the current Q network weights
        self._update_target_network()

        # Update the epsilon with the epsilon decay (and check that it doesn't go below 0)
        self.epsilon = self.initial_epsilon - self.epsilon_decay * self.current_epoch
        if self.epsilon < self.minimum_epsilon:
            self.epsilon = self.minimum_epsilon

        # Print the relevant info on the screen
        print("EPOCH " + str(self.current_epoch) + " FINISHED (Lines: " + str(lines) + "/Score: " + str(score) + "/Actions: " +  str(self.actions_performed) +"/Displacements: " + str(self.displacements) + ")")

        # Store the info for the current epoch into a CSV
        with open(join("results", self.class_name, self.folder_name, self.class_name + "_" + self.folder_name + "_data.csv"), 'a', newline='') as file:
            # Create the writer
            writer = csv.writer(file)
            # Insert the data
            writer.writerow([self.current_epoch, score, lines, self.displacements])

        # Store the weights into a folder (only every 10 epochs, to save size)
        if self.current_epoch % 10 == 0:
            self.q_network.save_weights(join("results", self.class_name, self.folder_name, "weights", self.class_name + "_" + self.folder_name +  "_epoch" +
                                             str(self.current_epoch) + ".h5"))

        # Reset the action counter
        self.actions_performed = 0

        # Update the epoch
        self.current_epoch += 1

