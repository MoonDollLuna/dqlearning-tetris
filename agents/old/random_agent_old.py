# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

# General imports
import sys
import numpy as np


class RandomAgentOld:
    """
    This class represents a random agent, used as a baseline.

    This agent is only intended for Play method, since it is not capable of any learning. It will return a random
    action every time it is asked.

    Therefore, all methods related to learning have been modified to instead close the program instantly
    (to avoid any unexpected error)
    """

    def __init__(self, seed):
        """
        Constructor of the class. Instantiates all necessary values.
        """

        # Create a dictionary to link every output of the agent to an actual action
        self.actions = {
            0: 'right',
            1: 'left',
            2: 'rotate',
            3: 'hard_drop'
        }

        # Store the seed and set it, if it has been provided.
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Mark this agent as an OLD agent (using the original implementation, where ACTION = user input)
        self.agent_type = "old"

        # Track the number of actions performed
        self.actions_performed = 0

    # Public methods

    def return_version(self):
        """
        Returns the version. Used to distinguish between agents of the old and new implementations.
        :return: the type of the agent
        """

        return self.agent_type

    def act(self, state):
        """
        For the current state, return a random action

        :param state: The current state provided by the game. Will be ignored.
        :return: (action, None)
                 WHERE
                 action: The action taken by the agent
                 None: Here, a real agent would return the Q-Values. In this case, since we have no actual network,
                 no values are returned
        """

        # Count the action taken
        self.actions_performed += 1

        # Take a random action from the actions dictionary
        # The strings are directly sampled in this case
        return np.random.choice(list(self.actions.values())), None

    def load_weights(self, weights):
        """
        This method exists only for compatibility with other agents.
        No weights can be loaded, since the agent does not contain a neural network
        """
        pass

    # All of the following methods exist only for compatibility sake, but if invoked they will only display
    # an error message and stop the execution (to avoid unexpected problems)

    def insert_experience(self, state, action, reward, next_state, terminated):
        print("ERROR: The random agent is not compatible with Learn mode. It can only be used in Play mode.")
        sys.exit()

    def initialize_learning_structure(self):
        print("ERROR: The random agent is not compatible with Learn mode. It can only be used in Play mode.")
        sys.exit()

    def finish_epoch(self, lines, score):
        print("ERROR: The random agent is not compatible with Learn mode. It can only be used in Play mode.")
        sys.exit()
