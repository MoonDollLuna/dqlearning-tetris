# CODE BY: Luna Jiménez Fernández

###########
# IMPORTS #
###########

# General imports
import sys
import numpy as np
import math


class ElTetrisAgent:
    """
    This class represents a deterministic agent, using the El-Tetris algorithm.

    This algorithm assigns a score to each state, using the following metrics:
    * Landing height of the piece
    * Rows eliminated
    * Row transitions
    * Column transitions
    * Number of holes
    * Number of wells

    More details about the algorithm itself can be found in the following direction:
    https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/

    This method is fully deterministic, and cannot be trained.
    Therefore, it can only be used in Play mode
    """

    def __init__(self):
        """
        Constructor of the class. Instantiates all necessary values.
        """
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
        For the current state, returns the action with the best El-Tetris score

        :param actions: A list of all possible actions, with structure:
                 (x_position, rotation, state)
        :return: action_taken, NONE
                 WHERE
                 action_taken: The action taken, passed as (x_position, rotation, state)
        """

        # Count the action taken
        self.actions_performed += 1

        # Initial placeholder action
        chosen_action = None
        chosen_score = float('-inf')

        # Evaluate every action
        for action in actions:
            action_score = self.eltetris_score(action)
            # If the action is better, choose it
            if action_score > chosen_score:
                chosen_action = action
                chosen_score = action_score

        # Return the best action
        return chosen_action, "None"

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

    # Score methods
    def eltetris_score(self, action):
        """
        Computes the El-Tetris score of a specific action

        The formula for the El-Tetris score is, specifically (approximating):
        -4.5 * landing_height +
        +3.41 * complete_lines +
        -3.22 * row_transitions +
        -9.34 * column_transitions +
        -7.89 * holes +
        -3.38 * well_sums
        """

        # Get the state and the column
        column = action[0]
        state = action[2]

        return (-4.500158825082766 * self.get_landing_height(column, state) +
                3.4181268101392694 * self.get_complete_lines(state) +
                -3.2178882868487753 * self.get_row_transitions(state) +
                -9.348695305445199 * self.get_column_transitions(state) +
                -7.899265427351652 * self.get_holes(state) +
                -3.3855972247263626 * self.get_wells(state))

    def get_landing_height(self, column, state):
        """
        Obtains the landing height of the piece.

        Since we're only able to see the state, this value will be approximated as
        the highest piece within the action column
        """

        # Convert the state to a numpy state and transpose it
        state = np.array(state)
        state = state.transpose()

        # Loop through the column
        depth = 19
        for position in state[column]:
            # Check if the position is filled
            if position == 1:
                break
            # If not, decrease the height
            else:
                depth -= 1

        # Return the final depth
        return depth

    def get_complete_lines(self, state):
        """
        Computes how many lines are fully complete (no holes)
        """

        # Convert the state to numpy
        state = np.array(state)

        # Compute which rows do not have holes
        full_rows = np.all(state != 0, axis=1)
        return np.sum(full_rows)

    def get_row_transitions(self, state):
        """
        Computes the row transitions in the state.

        A row transition happens when an empty cell is adjacent to a filled cell in the same row
        (and viceversa)
        """

        # Count the transitions
        transitions = 0

        # Loop through all rows
        for row in state:
            # Get the initial value of the row
            current_value = row[0]
            # Loop through the row
            for element in row:
                # If the element is different from the current value, a transition has happened
                if element != current_value:
                    current_value = element
                    transitions += 1

        return transitions

    def get_column_transitions(self, state):
        """
        Computes the column transitions in the state.

        A column transition happens when an empty cell is adjacent to a filled cell in the same column
        (and viceversa)
        """

        # Count the transitions
        transitions = 0

        # Generate the transposed state (to better access the columns)
        state = np.array(state)
        state = state.transpose()

        # Loop through all columns
        for column in state:
            # Get the initial value of the column
            current_value = column[0]
            # Loop through the column
            for element in column:
                # If the element is different from the current value, a transition has happened
                if element != current_value:
                    current_value = element
                    transitions += 1

        return transitions

    def get_holes(self, state):
        """
        Computes the total number of holes in the game state
        """

        holes = 0

        # Generate the numpy state
        state = np.array(state)

        # Get the dimensions of the state
        dimensions = state.shape
        # Loop by column, and then by row inside that column (from the bottom up)
        for x in range(dimensions[1]):
            # Store if we have found empty space (possible hole)
            empty_space = False

            for y in range(dimensions[0] - 1, -1, -1):

                # Currently not on a possible hole, but we find an empty space: mark a possible hole
                if not empty_space and state[y, x] == 0:
                    empty_space = True
                # On a possible hole and we find a locked piece in the state: hole confirmed, add it
                elif empty_space and state[y, x] != 0:
                    empty_space = False
                    holes += 1

        return holes

    def get_wells(self, state):
        """
        Computes the total number of wells in the game state

        A well is a succession of empty cells, such that the initial cells' left and right cells are filled

        The well value increases following the sequence 1 + 2 + 3...
        This way, deeper wells are worth more
        """

        # Well score
        wells = 0

        # Check for possible center wells (ignoring the sides)
        for x in range(1, 9):
            # Check, from the top down, for the start of a well
            for y in range(0, 20):
                if state[y][x] == 0 and state[y][x-1] == 1 and state[y][x+1] == 1:

                    # Well found
                    wells += 1

                    # Current well depth
                    well_depth = 1

                    # Check if the well continues
                    for new_y in range(y + 1, 20):
                        if state[new_y][x] == 0:
                            # Well continues
                            wells += well_depth + 1
                            well_depth += 1
                        else:
                            # Well is finished
                            break
                    break

            # Check for wells on the left side
            # Check, from the top down, for the start of a well
            for y in range(0, 20):
                if state[y][0] == 0 and state[y][1] == 1:

                    # Well found
                    wells += 1

                    # Current well depth
                    well_depth = 1

                    # Check if the well continues
                    for new_y in range(y + 1, 20):
                        if state[new_y][0] == 0:
                            # Well continues
                            wells += well_depth + 1
                            well_depth += 1
                        else:
                            # Well is finished
                            break
                    break

            # Check for wells on the right side
            # Check, from the top down, for the start of a well
            for y in range(0, 20):
                if state[y][9] == 0 and state[y][8] == 1:

                    # Well found
                    wells += 1

                    # Current well depth
                    well_depth = 1

                    # Check if the well continues
                    for new_y in range(y + 1, 20):
                        if state[new_y][9] == 0:
                            # Well continues
                            wells += well_depth + 1
                            well_depth += 1
                        else:
                            # Well is finished
                            break
                    break

        # Return the well score
        return wells