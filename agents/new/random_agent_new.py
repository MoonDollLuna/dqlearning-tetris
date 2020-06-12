# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

# General imports
import sys
import numpy as np


class RandomAgentNew:
    """
    This class represents a random agent, used as a baseline.

    This agent is only intended for Play method, since it is not capable of any learning. It will return a random
    action every time it is asked.

    Therefore, all methods related to learning have been modified to instead close the program instantly
    (to avoid any unexpected error)

    This agent works with the new action approach (where an action is the final position and rotation of the piece)
    """

    def __init__(self, seed):
        """
        Constructor of the class. Instantiates all necessary values.
        """

        # Store the seed and set it, if it has been provided.
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Mark this agent as an NEW agent (using the new implementation, where ACTION = final position and rotation
        # of the piece)
        self.agent_type = "new"

        # Track the number of actions performed
        self.actions_performed = 0

        # Keep track of the steps taken (displacements to the piece)
        self.displacements = 0

    # Public methods

    def return_version(self):
        """
        Returns the version. Used to distinguish between agents of the old and new implementations.
        :return: the type of the agent
        """

        return self.agent_type

    def act(self, actions):
        """
        For the current state, return a random action

        :param actions: A list of all possible actions, with structure:
                 (x_position, rotation, state)
        :return: action_taken, NONE
                 WHERE
                 action_taken: The action taken, passed as (x_position, rotation, state)
        """

        # Count the action taken
        self.actions_performed += 1

        # Select a random action from the actions list
        return np.random.choice(actions), None

    def load_weights(self, weights):
        """
        This method exists only for compatibility with other agents.
        No weights can be loaded, since the agent does not contain a neural network
        """
        pass

    def notify_step(self):
        """
        Increases the counter of steps (displacements). Called by the loop every time a step is taken
        """

        self.displacements += 1

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
