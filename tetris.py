# CODE BY: Luna Jiménez Fernández
# Originally based on the following tutorial:
# https://techwithtim.net/tutorials/game-development-with-python/tetris-pygame/tutorial-1/

# This file contains the full implementation of the Tetris game, as well as several methods used to assist the agents.
# The structure of the file is the following (in order)
#       1 - Imports
#       2 - Global variables
#           2A - Graphical variables
#           2B - Sound variables
#           2C - Gameplay variables
#           2D - AI-related variables
#       3 - Player variables
#       4 - Class definitions
#       5 - Graphical methods
#       6 - Sound methods
#       7 - Gameplay methods
#       8 - Agent and AI-related methods
#           8A - Graphical methods
#           8B - Gameplay methods
#       9 - Main loop auxiliary methods (mostly methods shared by all loops)
#       10 - Main loop methods
#           10A - Player main loop
#           10B - "Old" approach methods
#               10Ba - AI Player main loop
#               10Bb - AI Learner main loop
#           10C - "New" approach methods
#               10Ca - AI Player main loop
#               10Cb - AI Learner main loop
#           10D - Main menu logic
#       11 - Main method
#           11A - Argument definition
#           11B - Argument parsing
#           11C - Main logic

# The game was developed using PyGame.
# The code is fully documented, explaining inputs and outputs, and the use of every method.

# The game consists of a standard Tetris game, with a 10 x 20 grid (standard tetris size)
# The game also contains some specific choices, to aid with learning:
#       * Bag randomizer (a bag of all seven tetraminos is created, and random pieces are pulled from the bag. Once
#         the bag is empty, a new bag is generated). This ensures fairness in the random generation
#       * Instant lock once a piece stops on top of another piece (cannot move it to the sides or rotate)
#       * No capability to hold a piece (simpler to learn)
#       * (ONLY WHEN USING AI) Game speed is fixed (equal to agent polling speed). Eases the relation between action -> new state

###########
# IMPORTS #
###########

import sys
import argparse
import copy
import random
import os.path
from collections import deque

import numpy as np


# Import used for our own agent scripts
from agents.old import dql_agent_old, weighted_agent_old, random_agent_old
from agents.new import dql_agent_new, prioritized_agent_new, random_agent_new

import pygame


####################
# GLOBAL VARIABLES #
####################

# GRAPHICAL RELATED VARIABLES #

# Window size
screen_width = 700
screen_height = 800

# Extra window size when AI mode is active
screen_width_extra = 300

# Path to the custom font used
font_path = os.path.join(".", "fonts", "ARCADE_N.ttf")

# Block size
block_size = 40

# Playzone size
play_width = 10 * block_size # 10 block-wide playzone
play_height = 20 * block_size # 20 block-high playzone

# Playzone position
top_left_x = 20 + block_size // 2
top_left_y = 0

# Used colors
shape_colors = [(0, 240, 0), (240, 0, 0), (0, 240, 240), (240, 240, 0), (240, 160, 0), (0, 0, 240), (160, 0, 240)]
background_color = (170, 170, 170)
piece_border_color = (0, 0, 0)
playground_border_color = (75, 75, 75)
clear_color = (240, 240, 240)

# SOUND RELATED VARIABLES #

# Sound gallery
sound_gallery = {}

# Path to the background song
path_song = os.path.join(".", "sounds", "tetristhemea.mp3")

# GAMEPLAY RELATED VARIABLES

# Tetramino representation

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '0000.',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# List with all the shapes
shapes = [S, Z, I, O, J, L, T]

# Initial speed of the game (time between automatic piece fall, in milliseconds)
initial_speed = 500

# Speed modifier (how much the initial speed decreases each level increment, in milliseconds)
speed_modifier = 25

# Minimum speed (the maximum speed at which the game goes, in milliseconds)
# Typically, speed increases up to 9 times
minimum_speed = initial_speed - speed_modifier * 9

# LEARNING AND AI RELATED VARIABLES #

# Instantiated agent
# (Stored as a global variable, so it survives through loops and can always be accessed)
agent = None

# Polling speed (how often the agent acts, in milliseconds)
# Currently, the agent acts four times per second
# Initial speed will be adjusted to the same as the polling speed when an AI is active (simplification)
polling_speed = 250

# Initial speed when using an AI. This speed will NOT be modified with lines cleared. This is done to simplify the game
# and ensure that the agent movement and piece drops are always synchronized
game_speed_ai = polling_speed

# Maximum number of lines (if an agent reaches this amount of lines in a game while learning, the game is cut)
max_lines_training = 100

####################
# PLAYER VARIABLES #
####################

# All of these variables can be set using arguments while launching the script
# However, they're stored here to give them default values

# Whether the sound is active (TRUE) or not (FALSE)
# Set to false using --silent
sound_active = True

# Whether FIXED SPEED mode is active (TRUE) or not (FALSE). If fixed speed is active, the speed is fixed and will not
# increase with the difficulty level. Also, the scoring system used will be the same as the AI one.
# Set to true using --fixedspeed
fixed_speed = False

# If the player is human (FALSE) or an AI (TRUE)
# Set using --ai play or --ai learn
ai_player = False

# If the AI player is in playing (FALSE) or learning (TRUE) mode
# Set using --ai learn
ai_learning = False

# Set seed for reproducibility. NONE if no seed has been set
# Set using --seed
seed = None

# Set type of agent to be used. Default value is Standard (Old). Only relevant when using AI (training or playing)
# Existing agent types:
#   * Standard (new): Standard DQL Agent using the new approach (action = final position of the piece)
#   * Prioritized (new): Variation of the Standard (new) agent using Prioritized Experience Replay (PER)
#   * Random (new): Agent exclusively used in Play mode. Acts randomly, serves as a baseline
#   * Standard (old): Standard DQL Agent using the original approach (action = player input)
#   * Weighted (old): Variation of the Standard (old) agent using weights for the actions when randomly choosing them
#   * Random (old): Agent exclusively used in Play mode. Acts randomly, serves as a baseline
# Set using --agenttype
agent_type = 'standard_new'

# Loaded weights for the AI Player. Only used when there is an AI player active (not while learning). Default value is
# None if not specified
# Set using --weights
weights = None

# TRAINING VARIABLES - These variables are only relevant while ai_learning is TRUE #

# If the training is being done in normal, visual mode (FALSE) or in fast, text-only mode (TRUE)
# Set using --fast
fast_training = False

# Maximum amount of elements contained within the Experience Replay. Default value is specified below
# Set using --experiencereplaysize
experience_replay_size = 20000

# Batch size used to sample from the Experience Replay. Default value is specified below
# Set using --batchsize
batch_size = 32

# Method used to compute the rewards. There are two possible values:
# * 'game' for a method based directly on the game score
# * 'heuristic' for a method based on scoring the state according to a heuristic function
# Set using --rewards
rewards_method = 'game'

# Gamma value used by DQL (learning rate of DQL). Default value is specified below
# Set using --gamma
gamma = 0.99

# Epsilon value used by DQL (chance to perform a random action in exploration-exploitation)
# Default value is specified below
# Set using --epsilon
epsilon = 1

# Epsilon percentage for minimum used by DQL (decrease the epsilon value linearly every epoch until this percentage of
# epochs have been completed). After this percentage, epsilon = minimum_epsilon
# Set using --epsilonpercentage
epsilon_percentage = 75

# Minimum epsilon value used by DQL (the minimum value of epsilon, achieved after epsilon_percentage percent of epochs
# have passed)
# Set using --minimumepsilon
minimum_epsilon = 0.05

# Learning rate used by the neural network. Default value is specified below
# Set using --learningrate
learning_rate = 0.001

# Number of epochs used to train the agent. Default value is specified below.
# Set using --epochs
total_epochs = 2000


#####################
# CLASS DEFINITIONS #
#####################

class Piece(object):
    """Tetramino (piece) used by the game."""

    def __init__(self, x, y, shape):
        """
        Constructor. Creates a piece, indicating the (x, y) position and the specific shape.

        :param x: Initial x position of the shape.
        :param y: Initial y position of the shape.
        :param shape: Shape to be used.
        """

        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0


#####################
# GRAPHICAL METHODS #
#####################

def draw_manager(surface, grid, current_shape, next_shape, score=0, level=0, lines=0):
    """
    Draws all the elements on the screen.

    :param surface: Surface used to hold all elements.
    :param grid: Matrix containing the current state of the playzone.
    :param current_shape: Shape to be drawn with a shadow drop.
    :param grid: Grid containing all the fixed blocks.
    :param next_shape: Shape to be drawn on the NEXT SHAPE screen
    :param score: Current score of the player.
    :param level: Current level of the game.
    :param lines: Current lines cleared by the player.
    """

    # Fill the background with black
    surface.fill((15, 15, 15))

    # Draw the playzone
    draw_playzone(surface, grid)

    # Draws the shadow drop
    draw_shadow_drop(surface, current_shape, grid)

    # Draw the next piece
    draw_next_shape(surface, next_shape)

    # Draw the rest of the hud
    draw_hud(surface, score, level, lines)


def draw_playzone(surface, grid):
    """
    Draws the playzone (the cells and all the blocks on it)

    :param surface: Surface used to hold the playzone.
    :param grid: Matrix containing the current state of the playzone.
    """

    # For each block in the grid
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # Draw the block in the appropiate color
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)
            # If not background, draw the border
            if grid[i][j] != background_color:
                pygame.draw.rect(surface, piece_border_color, (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 1)
            # Else, draw a light gray grid
            else:
                pygame.draw.rect(surface, (160, 160, 160), (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 1)

    # Draws two borders around the playground
    pygame.draw.rect(surface, playground_border_color, (20, 0, block_size // 2, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (top_left_x + play_width, 0, block_size // 2, screen_height), 0)


def draw_shadow_drop(surface, shape, grid):
    """
    Draws the shadow drop of the current piece (where it would fall if you hard dropped it right now)

    :param surface: Surface to draw the shape on.
    :param shape: Shape to be drawn.
    :param grid: Grid containing all the fixed blocks.
    """

    # Clone the shape and the grid
    shadow_shape = copy.deepcopy(shape)
    clones_grid = copy.deepcopy(grid)

    # Remove the current piece from the grid
    for (x, y) in generate_shape_positions(shadow_shape):
        if y >= 0:
            clones_grid[y][x] = background_color

    # Find the position where the piece would be
    while valid_space(shadow_shape, clones_grid):
        shadow_shape.y += 1
    shadow_shape.y -= 1

    # Draw all the blocks currently not overlapping with the shape in the appropiate color
    for (x, y) in generate_shape_positions(shadow_shape):
        if (x, y) not in generate_shape_positions(shape):
            pygame.draw.rect(surface, shadow_shape.color, (top_left_x + x * block_size, top_left_y + y * block_size, block_size, block_size), 5)


def draw_next_shape(surface, shape):
    """
    Draws the next expected shape in the HUD

    :param surface: Surface to draw the shape on.
    :param shape: Shape to be drawn.
    """

    # Identify where to place the shapes
    hud_begin_x = top_left_x + play_width + 30
    hud_begin_y = 480

    # Draw a rectangle to contain the title
    pygame.draw.rect(surface, background_color, (hud_begin_x, hud_begin_y + 20, screen_width - hud_begin_x - 10, 60), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x, hud_begin_y + 20, screen_width - hud_begin_x - 10, 60), 5)

    # Draw the title and place it
    font = pygame.font.Font(font_path, 20)
    text = font.render('NEXT SHAPE', 1, (0, 0, 0))
    surface.blit(text, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - text.get_width() / 2, hud_begin_y + 50 - text.get_height() / 2))

    # Draw a rectangle to contain the shape below
    pygame.draw.rect(surface, background_color, (hud_begin_x + 10, hud_begin_y + 90, screen_width - hud_begin_x - 30, 180), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x + 10, hud_begin_y + 90, screen_width - hud_begin_x - 30, 180), 5)

    # Draw the shape in the box
    shape_format = shape.shape[shape.rotation % len(shape.shape)]

    sx = top_left_x + play_width + 40
    sy = hud_begin_y + 100

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*block_size, sy + i * block_size, block_size, block_size), 0)
                pygame.draw.rect(surface, piece_border_color, (sx + j * block_size, sy + i * block_size, block_size, block_size), 1)


def draw_hud(surface, score, level, lines):
    """
    Draw the rest of the elements of the HUD (score, level, lines)

    :param surface: Surface on which to draw the elements.
    :param score: Current score of the player.
    :param level: Current level of the player.
    :param lines: Current amount of cleared lines by the player
    """

    # Identify where to place the HUD
    hud_begin_x = top_left_x + play_width + 30

    # Create the font
    font = pygame.font.Font(font_path, 20)

    # SCORE
    score_y = 75

    # Score uses its own font
    score_font = pygame.font.Font(font_path, 30)

    # Draw the rectangle
    pygame.draw.rect(surface, background_color, (hud_begin_x - 10, score_y, screen_width - hud_begin_x + 10, 90), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x - 10, score_y, screen_width - hud_begin_x + 10, 90), 5)

    # Write the text and print it
    score_text = score_font.render('SCORE', 1, (0, 0, 0))
    score_score = font.render(str(score), 1, (0, 0, 0))

    surface.blit(score_text, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - score_text.get_width() / 2, score_y + 30 - score_text.get_height() / 2))
    surface.blit(score_score, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - score_score.get_width() / 2, score_y + 65 - score_score.get_height() / 2))

    # LEVEL
    level_y = 250
    # Draw the rectangle
    pygame.draw.rect(surface, background_color, (hud_begin_x, level_y, screen_width - hud_begin_x - 10, 80), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x, level_y, screen_width - hud_begin_x - 10, 80), 5)

    # Write the text and print it
    level_text = font.render('LEVEL', 1, (0, 0, 0))
    level_score = font.render(str(level), 1, (0, 0, 0))

    surface.blit(level_text, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - level_text.get_width() / 2, level_y + 25 - level_text.get_height() / 2))
    surface.blit(level_score, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - level_score.get_width() / 2, level_y + 55 - level_score.get_height() / 2))

    # LINES
    lines_y = 350
    # Draw the rectangle
    pygame.draw.rect(surface, background_color, (hud_begin_x, lines_y, screen_width - hud_begin_x - 10, 80), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x, lines_y, screen_width - hud_begin_x - 10, 80), 5)

    # Write the text and print it
    lines_text = font.render('LINES', 1, (0, 0, 0))
    lines_score = font.render(str(lines), 1, (0, 0, 0))

    surface.blit(lines_text, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - lines_text.get_width() / 2,
                              lines_y + 25 - lines_text.get_height() / 2))
    surface.blit(lines_score, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - lines_score.get_width() / 2,
                               lines_y + 55 - lines_score.get_height() / 2))


def draw_clear_row(surface, lines):
    """
    Draws an effect when a line is cleared.

    :param surface: Surface on which to draw the effect.
    :param lines: List containing all of the cleared lines.
    """

    # Draws the effect on all cleared lines
    for i in lines:
        for j in range(0, 10):
            pygame.draw.rect(surface, clear_color, (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)
        pygame.draw.rect(surface, clear_color, (top_left_x, top_left_y + i*block_size, play_width, block_size), 3)

    # Print the effect on the screen for a bit
    pygame.display.flip()
    pygame.time.wait(300)


def draw_game_over_effect(surface):
    """
    Draws an effect on the playzone when a line is cleared.

    :param surface: Surface on which to draw the effect
    """

    # Draws the effect on all lines (from the bottom row to the first one), waiting a bit on each one
    for i in range((play_height // block_size) - 1, -1, -1):
        for j in range(0, (play_width // block_size)):
            pygame.draw.rect(surface, (200, 200, 200), (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)
            pygame.draw.rect(surface, piece_border_color,(top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 1)
        pygame.display.flip()
        pygame.time.wait(40)

    pygame.time.wait(300)

    # Draws GAME OVER on the play zone
    font = pygame.font.Font(font_path, 40)
    text = font.render('GAME OVER!', 1, (15, 15, 15))
    surface.blit(text, (top_left_x + (play_width // 2) - text.get_width() / 2, screen_height // 2 - text.get_height() / 2))
    pygame.display.flip()

    # Waits a bit of extra time (for good measure)
    pygame.time.wait(1500)


def draw_main_menu(surface):
    """
    Draws the main menu.

    If AI mode is active, the menu will have some adjustments to ensure that everything is centered
    (since the screen is bigger in AI mode)

    :param surface: Surface on which to draw the menu.
    """

    # If AI is active, obtain the new window width
    if ai_player:
        screen_width_menu = screen_width + screen_width_extra
    else:
        screen_width_menu = screen_width

    # Draws the background
    surface.fill((15, 15, 15))

    # Creates the title
    title_font = pygame.font.Font(font_path, 100)
    title_text = title_font.render('TETRIS', 1, (255, 255, 255))
    surface.blit(title_text, (screen_width_menu / 2 - title_text.get_width() / 2, 150 - title_text.get_height() / 2))

    # Create the subtitle
    subtitle_font = pygame.font.Font(font_path, 30)
    subtitle_text = subtitle_font.render('FOR DEEP-Q LEARNING', 1, (255, 255, 255))
    surface.blit(subtitle_text, (screen_width_menu / 2 - subtitle_text.get_width() / 2, 225 - subtitle_text.get_height() / 2))

    # Create the start message
    start_font = pygame.font.Font(font_path, 30)
    start_text = start_font.render('PRESS ANY', 1, (255, 255, 255))
    surface.blit(start_text, (screen_width_menu / 2 - start_text.get_width() / 2, 450 - start_text.get_height() / 2))
    start_text2 = start_font.render('KEY TO START!', 1, (255, 255, 255))
    surface.blit(start_text2, (screen_width_menu / 2 - start_text2.get_width() / 2, 500 - start_text2.get_height() / 2))

    # Create the developed disclaimer
    developed_font = pygame.font.Font(font_path, 15)
    developed_text = developed_font.render('DEVELOPED BY LUNA JIMENEZ FERNANDEZ', 1, (255, 255, 255))
    surface.blit(developed_text, (screen_width_menu / 2 - developed_text.get_width() / 2, 750 - developed_text.get_height() / 2))

    # If the player is an AI, indicate it on the main screen
    if ai_player:
        ai_font = pygame.font.Font(font_path, 15)
        ai_text = ai_font.render('AI PLAYER ACTIVE', 1, (255, 255, 255))
        surface.blit(ai_text,
                     (screen_width_menu / 2 - ai_text.get_width() / 2, 625 - ai_text.get_height() / 2))

    # Draw the screen
    pygame.display.flip()


#################
# SOUND METHODS #
#################

def prepare_sounds(list_sounds):
    """
    Creates a sound gallery with all the initialized Sounds.

    :param list_sounds: List containing the name and path of all the sound files.
    """

    dictionary_sounds = {}

    # For each sound in path, initialize it
    for (name, path) in list_sounds:
        proper_path = os.path.join(".", "sounds", path)
        dictionary_sounds[name] = pygame.mixer.Sound(proper_path)

    return dictionary_sounds


def play_sound(sound):
    """
    If sound is active, play a sound.

    :param sound: Name of the sound to be played
    """

    if sound_active:
        if sound in sound_gallery:
            sound_gallery[sound].play()


def play_song():
    """Starts playing the background music on loop (if sound is active)."""

    if sound_active:
        pygame.mixer.music.load(path_song)
        pygame.mixer.music.play(-1)


def stop_sounds():
    """Stops playing all sounds."""

    pygame.mixer.pause()
    pygame.mixer.music.stop()


####################
# GAMEPLAY METHODS #
####################

def bag_randomizer():
    """
    Generates a random set of pieces. All seven pieces are included in a random order.
    :return: A randomized list of pieces.
    """

    # Get the list of pieces and randomly shuffle it
    shuffled_list = copy.deepcopy(shapes)
    random.shuffle(shuffled_list)

    return shuffled_list


def get_shape(shapes_list):
    """
    Gets a shape from the shapes list. If it is empty, refills it using a bag randomizer.

    :param shapes_list: List of shapes from which to get the shape.
    :return: The shape and the modified list of shapes.
    """

    # Check if the bag of pieces is empty
    if len(shapes_list) == 0:
        # If it is, refill it with the seven pieces (in a random order)
        shapes_list = bag_randomizer()

    # Take the top value from the list and create the shape
    shape = shapes_list.pop(0)
    piece = Piece(5, 0, shape)

    return piece, shapes_list


def create_grid(locked_positions={}):
    """
    Generates the grid (the inner codification of the playzone).

    :param locked_positions: A dictionary containing the position of all locked pieces (including the current piece).
    Key = (x, y) position of the piece. Value = Color of the piece.
    :return: Matrix containing all the colors of the grid, where matrix[y][x] = color in the position (x, y).
    """

    # Grid (playzone) is represented as a matrix of colours
    # Initially all colors (empty positions) are of the background color
    grid = [[background_color for _ in range(10)] for _ in range(20)]

    # For all fixed blocks (locked positions), color the corresponding position to the appropiate color
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                grid[i][j] = locked_positions[(j, i)]

    return grid


def generate_shape_positions(shape):
    """
    Converts the current shape position into a list of usable (x, y) coordinates.

    :param shape: Shape to obtain the position of.
    :return: List containing the (x, y) coordinates of all the blocks of the shape.
    """

    # List of positions to be returned
    positions = []

    # Obtain the codification of the current shape
    shape_format = shape.shape[shape.rotation % len(shape.shape)]

    # Explores the codification of the shape, line by line
    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            # If a 0 is found (a block), add the position to the list of positions
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    # Since the codifications have an offset, remove it to obtain the true position
    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    """
    Checks if the position of the shape would be valid in the current grid.

    :param shape: Shape to check if the position is valid.
    :param grid: Grid containing the state of the playzone.
    :return: True if the position is valid, False otherwise.
    """

    # Obtain a list with all the valid positions (positions that have the background color)
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == background_color] for i in range(20)]
    # Add an extra four layers on top (index from -4 to -1), to account for pieces above the playing area
    # This is added to ensure that pieces that just spawned can move only in legal positions
    extra_lines = [[(j, i) for j in range(10)] for i in range(-4, 0)]
    accepted_pos.extend(extra_lines)
    # Flatten everything
    accepted_pos = [j for sub in accepted_pos for j in sub]

    # Obtain the coordinates of the shape
    formatted = generate_shape_positions(shape)

    # Check if the coordinates of the shape are in the accepted position
    for pos in formatted:
        # Position invalid
        if pos not in accepted_pos:
            return False

    return True


def check_defeat(positions):
    """
    Check if the game is over (a piece has reached the top of the screen)

    :param positions: Dictionary of locked positions, where key = (x, y) position and value = color of the position.
    :return: True if the game is over, False otherwise.
    """

    # Loop over all the keys
    for pos in positions:
        # If any locked position has y < 1 (0 or greater), the top has been reached and the game is over
        if pos[1] < 1:
            return True
    return False


def clear_rows(grid, locked):
    """
    Removes all cleared rows from the grid.

    :param grid: Matrix representing the current state of the playzone.
    :param locked: Dictionary of locked positions, where Key = (x, y) coordinates of the piece.
    :return: List of cleared lines
    """

    # Removed lines
    removed_lines = []

    # Compute which lines to remove
    # For all values from first to the last row
    for i in range(0, len(grid)):
        row = grid[i]
        # If there is no background color (the row is full of blocks)
        if background_color not in row:
            # Add the current row to the list of removed rows
            removed_lines.append(i)

            # Remove all blocks in the line
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue

    # If lines have been removed, update the necessary lines in the grid with the new y values
    if len(removed_lines) > 0:
        # Explored in reverse order
        # (otherwise, we would crush previous positions)

        # Note: [::-1] is an iterator, that basically allows to access the sorted list in an inverse order
        # (from the biggest to the smallest value)
        # Iterators use the format [start:end:step]
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key

            # Compute the offset of the line
            offset = 0
            for line in removed_lines:
                if line > y:
                    offset += 1

            # Update the value of blocks
            new_key = (x, y + offset)
            locked[new_key] = locked.pop(key)

    return removed_lines


def compute_score(lines_cleared, lowest_y, level):
    """
    Obtains the actual score from the amount of cleared lines, taking into account the lowest Y position and the
    difficulty level.

    SCORE:
    * lowest_y * (level + 1) if no lines have been cleared
    * 100 * 2^(lines_cleared - 1) * (level + 1) if lines have been cleared (so 100, 200, 400 or 800 points)

    If the game is in AI mode or the fixed speed flag is set, the score is simplified:
    * 1 if no lines have been cleared
    * 10 * 2^(lines_cleared-1) if lines have been cleared (so 10, 20, 40 or 80 points)

    :return: Computed score
    """

    # Check if AI or fixed speed is active
    if ai_player or fixed_speed:
        # ACTIVE: Use special scoring system
        if lines_cleared == 0:
            # No lines cleared
            return 1
        else:
            # Lines cleared
            return 10 * (2 ** (lines_cleared - 1))
    else:
        # INACTIVE: Use the standard scoring system
        if lines_cleared == 0:
            # No lines cleared
            return lowest_y * (level + 1)
        else:
            # Lines cleared
            return 100 * (2 ** (lines_cleared - 1)) * (level + 1)


#################
# AGENT METHODS #
#################

# GRAPHICS #

def draw_state(surface, state, initial_x, initial_y, grid_block_size):
    """
    Draws the state starting in the specified (x, y) position.

    The state will be drawn as a small grid, with each cell having the specified color:
    * BLACK if the cell is empty
    * WHITE if the cell is occupied (either by a locked piece or by the current piece)

    :param surface: Surface in which to draw everything
    :param state: Current state, already processed (20x10 matrix)
    :param initial_x: X position of the top left corner
    :param initial_y: Y position of the top left corner
    :param grid_block_size: Size (in pixels) of each cell of the grid
    """

    # Draw a rectangle behind the state
    pygame.draw.rect(surface, playground_border_color,
                     (initial_x - 5, initial_y - 5, grid_block_size * 10 + 10, grid_block_size * 20 + 10), 0)

    # Loop through all elements of the state
    for (y, x), element in np.ndenumerate(state):

        # Set the color depending on the value of the element
        # (0 is black, 1 is white)
        if element == 0:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)

        # Draw the actual square in the appropiate position
        pygame.draw.rect(surface,
                         color,
                         (initial_x + x * grid_block_size,
                          initial_y + y * grid_block_size,
                          grid_block_size,
                          grid_block_size),
                         0)


def draw_ai_player_old_information(surface, current_state, q_values, action, actions_taken):
    """
    Draws relevant information for the AI player (in order to visualize the choices being taken)

    :param surface: Surface in which to draw the information.
    :param current_state: Last state received by the agent, to be displayed on screen
    :param q_values: Q-Value of every action (can be None)
    :param action: Last action taken
    :param actions_taken: Total number of actions taken
    """

    # Create a rectangle for the additional HUD
    pygame.draw.rect(surface, background_color, (screen_width, 0, screen_width_extra, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (screen_width, 0, screen_width_extra, screen_height), 5)

    # Identify where to place the additional AI HUD
    hud_begin_x = screen_width + 25

    # Create all necessary fonts
    small_font = pygame.font.Font(font_path, 10)
    big_font = pygame.font.Font(font_path, 20)

    # STATE

    # Title
    state_y = 15

    # Write the state title and print it
    state_text = big_font.render('PASSED STATE:', 1, (0, 0, 0))
    surface.blit(state_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - state_text.get_width() / 2,
                              state_y + 25 - state_text.get_height() / 2))

    # Draw the state itself
    draw_state(win, current_state, screen_width + 75, 70, 15)

    # If Q-Values are present, draw all the relevant information
    if q_values is not None:
        # ACTION Q-VALUES:
        qvalues_y = 380

        # Draw a black rectangle for all Q values
        pygame.draw.rect(surface,
                         playground_border_color,
                         (screen_width,
                          qvalues_y,
                          screen_width_extra,
                          300),
                         0)

        # Draw the title
        qvalues_text = big_font.render('Q-VALUES', 1, (0, 0, 0))
        surface.blit(qvalues_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - qvalues_text.get_width() / 2, qvalues_y + 25 - qvalues_text.get_height() / 2))

        # Find the action with the biggest Q-Value
        best_action = np.argmax(q_values[0])

        # Draw the text for each action and its Q-Value
        # The order followed is respected by the agent
        # If the text has the highest Q value, a rectangle will be drawn behind

        # Draw a rectangle for the best action
        pygame.draw.rect(surface,
                         background_color,
                         (screen_width,
                          qvalues_y + 50 + best_action * 60,
                          screen_width_extra,
                          60),
                         0)

        # RIGHT
        right_text = big_font.render('RIGHT:', 1, (0, 0, 0))
        surface.blit(right_text, (hud_begin_x, qvalues_y + 50))

        # Right content
        right_content_text = big_font.render(str(q_values[0][0]), 1, (0, 0, 0))
        surface.blit(right_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - right_content_text.get_width() / 2, qvalues_y + 80))

        # LEFT
        left_text = big_font.render('LEFT:', 1, (0, 0, 0))
        surface.blit(left_text, (hud_begin_x, qvalues_y + 110))

        # Left content
        left_content_text = big_font.render(str(q_values[0][1]), 1, (0, 0, 0))
        surface.blit(left_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - left_content_text.get_width() / 2, qvalues_y + 140))

        # ROTATE
        rotate_text = big_font.render('ROTATE:', 1, (0, 0, 0))
        surface.blit(rotate_text, (hud_begin_x, qvalues_y + 170))

        # Rotate content
        rotate_content_text = big_font.render(str(q_values[0][2]), 1, (0, 0, 0))
        surface.blit(rotate_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - rotate_content_text.get_width() / 2, qvalues_y + 200))

        # HARD DROP
        harddrop_text = big_font.render('HARD DROP:', 1, (0, 0, 0))
        surface.blit(harddrop_text, (hud_begin_x, qvalues_y + 230))

        # Rotate content
        harddrop_content_text = big_font.render(str(q_values[0][3]), 1, (0, 0, 0))
        surface.blit(harddrop_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - harddrop_content_text.get_width() / 2, qvalues_y + 260))

    # LAST ACTION
    action_y = 700
    action_text = small_font.render('ACTION TAKEN: ' + action, 1, (0, 0, 0))
    surface.blit(action_text, (hud_begin_x, action_y))

    # TOTAL ACTIONS TAKEN
    total_actions_y = 730
    total_actions_text = small_font.render('TOTAL ACTIONS TAKEN: ' + str(actions_taken), 1, (0, 0, 0))
    surface.blit(total_actions_text, (hud_begin_x, total_actions_y))


def draw_ai_learn_old_information(surface, current_state, next_state, action, reward, current_epoch, actions_performed,
                                  current_epsilon, best_epoch, best_score, best_lines, best_actions_performed):
    """
    Draws relevant information for the AI learner (in order to visualize the inner workings)

    :param surface: Surface in which to draw the information.
    :param current_state: Original state received by the agent, in order to visualize it
    :param next_state: State reached from the original state, in order to visualize it
    :param action: Action taken in the current state
    :param reward: Reward obtained for the pair state/action
    :param current_epoch: Current epoch of training
    :param actions_performed: Total number of actions performed
    :param current_epsilon: Current epsilon value for the agent
    :param best_epoch: Best epoch of training
    :param best_score: Score from the best epoch
    :param best_lines: Lines cleared from the best epoch
    :param best_actions_performed: Total number of actions performed in the best epoch
    """

    # Create a rectangle for the additional HUD
    pygame.draw.rect(surface, background_color, (screen_width, 0, screen_width_extra, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (screen_width, 0, screen_width_extra, screen_height), 5)

    # Identify where to place the additional AI HUD
    hud_begin_x = screen_width + 25

    # Create necessary fonts
    small_font = pygame.font.Font(font_path, 15)
    tiny_font = pygame.font.Font(font_path, 10)

    # CURRENT EPOCH TITLE
    epoch_title_y = 15
    epoch_title_text = small_font.render('CURRENT EPOCH: ' + str(current_epoch), 1, (0, 0, 0))
    surface.blit(epoch_title_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - epoch_title_text.get_width() / 2,
                                    epoch_title_y + 25 - epoch_title_text.get_height() / 2))

    # STATES

    # Title height
    state_x = hud_begin_x + 15
    state_y = 50
    # Write the state titles and print them
    current_state_text = small_font.render('CURRENT', 1, (0, 0, 0))
    surface.blit(current_state_text, (state_x, state_y + 25 - current_state_text.get_height() / 2))

    new_state_text = small_font.render('NEXT', 1, (0, 0, 0))
    surface.blit(new_state_text, (state_x + 150, state_y + 25 - current_state_text.get_height() / 2))

    # Draw the states themselves
    draw_state(surface, current_state, state_x, 100, 10)
    draw_state(surface, next_state, state_x + 120, 100, 10)

    # ACTION, REWARD AND TOTAL ACTIONS TAKEN
    action_y = 350
    action_text = tiny_font.render('ACTION TAKEN: ' + action, 1, (0, 0, 0))
    surface.blit(action_text, (hud_begin_x, action_y))

    reward_y = 380
    reward_text = tiny_font.render('REWARD: ' + str(reward), 1, (0, 0, 0))
    surface.blit(reward_text, (hud_begin_x, reward_y))

    actions_y = 410
    actions_text = tiny_font.render('ACTIONS TAKEN: ' + str(actions_performed), 1, (0, 0, 0))
    surface.blit(actions_text, (hud_begin_x, actions_y))

    # CURRENT EPSILON

    epsilon_y = 500
    epsilon_text = tiny_font.render('CURRENT EPSILON: ' + str(current_epsilon), 1, (0, 0, 0))
    surface.blit(epsilon_text, (hud_begin_x, epsilon_y))

    # BEST EPOCH
    best_epoch_title_y = 550
    best_epoch_title_text = small_font.render('BEST EPOCH: ' + str(best_epoch), 1, (0, 0, 0))
    surface.blit(best_epoch_title_text, (
    hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - best_epoch_title_text.get_width() / 2,
    best_epoch_title_y + 25 - best_epoch_title_text.get_height() / 2))

    best_epoch_lines_y = 600
    best_epoch_lines_text = tiny_font.render('LINES: ' + str(best_lines), 1, (0, 0, 0))
    surface.blit(best_epoch_lines_text, (hud_begin_x, best_epoch_lines_y))

    best_epoch_score_y = 630
    best_epoch_score_text = tiny_font.render('SCORE: ' + str(best_score), 1, (0, 0, 0))
    surface.blit(best_epoch_score_text, (hud_begin_x, best_epoch_score_y))

    best_epoch_actions_y = 660
    best_epoch_actions_text = tiny_font.render('ACTIONS TAKEN: ' + str(best_actions_performed), 1, (0, 0, 0))
    surface.blit(best_epoch_actions_text, (hud_begin_x, best_epoch_actions_y))


def draw_ai_player_new_information(surface, target_state, q_value, last_step, actions_taken, steps_taken):
    """
    Draws relevant information for the AI player (in order to visualize the choices being taken)
    This method is used only for "new" approach AI players

    :param surface: Surface in which to draw the information
    :param target_state: State to be reached by moving the piece
    :param q_value: Q-Value of the chosen action
    :param last_step: Last step taken by the agent
    :param actions_taken: Total number of actions taken
    :param steps_taken: Total number of steps taken
    """

    # Create a rectangle for the additional HUD
    pygame.draw.rect(surface, background_color, (screen_width, 0, screen_width_extra, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (screen_width, 0, screen_width_extra, screen_height), 5)

    # Identify where to place the additional AI HUD
    hud_begin_x = screen_width + 25

    # Create all necessary fonts
    small_font = pygame.font.Font(font_path, 10)
    big_font = pygame.font.Font(font_path, 15)

    # AGENT BEING USED
    agent_y = 15
    agent_text = small_font.render('AGENT: ' + agent.__class__.__name__, 1, (0, 0, 0))
    surface.blit(agent_text, (hud_begin_x, agent_y))

    # TARGET ACTION

    # Title
    target_y = 40

    # Write the state title and print it
    target_text = big_font.render('GOAL STATE:', 1, (0, 0, 0))
    surface.blit(target_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - target_text.get_width() / 2,
                               target_y + 25 - target_text.get_height() / 2))

    # Draw the state itself
    draw_state(win, target_state, screen_width + 75, 100, 15)

    # Q-VALUE OF THE STATE
    q_value_y = 450
    q_value_text = small_font.render('Q-VALUE: ' + str(q_value[0]), 1, (0, 0, 0))
    surface.blit(q_value_text, (hud_begin_x, q_value_y))

    # LAST STEP
    step_y = 550
    step_text = small_font.render('LAST STEP: ' + last_step, 1, (0, 0, 0))
    surface.blit(step_text, (hud_begin_x, step_y))

    # TOTAL ACTIONS AND STEPS TAKEN
    total_actions_y = 700
    total_actions_text = small_font.render('TOTAL ACTIONS TAKEN: ' + str(actions_taken), 1, (0, 0, 0))
    surface.blit(total_actions_text, (hud_begin_x, total_actions_y))

    total_steps_y = 740
    total_steps_text = small_font.render('TOTAL STEPS TAKEN: ' + str(steps_taken), 1, (0, 0, 0))
    surface.blit(total_steps_text, (hud_begin_x, total_steps_y))


def draw_ai_learn_new_information(surface, current_epoch, original_state, goal_state, q_value, actions_taken, steps_taken,
                                  current_epsilon, best_epoch, best_lines, best_score, best_actions, best_steps):
    """
    Draws relevant information for the AI learner (in order to visualize the inner workings)

    :param surface: Surface in which to draw the information
    :param current_epoch: Current epoch
    :param original_state: Original state received by the agent, in order to visualize it
    :param goal_state: State reached from the original state, in order to visualize it
    :param q_value: Q-Value for the goal state
    :param actions_taken: Total amounts of actions performed in this epoch
    :param steps_taken: Total amount of steps (piece movements) performed in this epoch
    :param current_epsilon: Value of epsilon in this epoch
    :param best_epoch: Epoch where the best values were found
    :param best_score: Amount of lines cleared during the best epoch
    :param best_lines: Score achieved during the best epoch
    :param best_actions: Actions performed during the best epoch
    :param best_steps: Steps performed during the best epoch
    """

    # Create a rectangle for the additional HUD
    pygame.draw.rect(surface, background_color, (screen_width, 0, screen_width_extra, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (screen_width, 0, screen_width_extra, screen_height), 5)

    # Identify where to place the additional AI HUD
    hud_begin_x = screen_width + 25

    # Create necessary fonts
    small_font = pygame.font.Font(font_path, 15)
    tiny_font = pygame.font.Font(font_path, 10)

    # AGENT BEING USED
    agent_y = 15
    agent_text = tiny_font.render('AGENT: ' + agent.__class__.__name__, 1, (0, 0, 0))
    surface.blit(agent_text, (hud_begin_x, agent_y))

    # CURRENT EPOCH TITLE
    epoch_title_y = 40
    epoch_title_text = small_font.render('CURRENT EPOCH: ' + str(current_epoch), 1, (0, 0, 0))
    surface.blit(epoch_title_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - epoch_title_text.get_width() / 2,
                                    epoch_title_y + 25 - epoch_title_text.get_height() / 2))

    # STATES

    # Title height
    state_x = hud_begin_x + 15
    state_y = 70
    # Write the state titles and print them
    original_state_text = small_font.render('ORIGINAL', 1, (0, 0, 0))
    surface.blit(original_state_text, (state_x, state_y + 25 - original_state_text.get_height() / 2))

    goal_state_text = small_font.render('GOAL', 1, (0, 0, 0))
    surface.blit(goal_state_text, (state_x + 150, state_y + 25 - goal_state_text.get_height() / 2))

    # Draw the states themselves
    draw_state(surface, original_state, state_x, 130, 10)
    draw_state(surface, goal_state, state_x + 120, 130, 10)

    #Q VALUE
    q_value_y = 350
    q_value_text = tiny_font.render('Q-VALUE: ' + str(q_value[0]), 1, (0, 0, 0))
    surface.blit(q_value_text, (hud_begin_x, q_value_y))

    # ACTIONS AND STEPS
    actions_y = 450
    actions_text = tiny_font.render('ACTIONS TAKEN: ' + str(actions_taken), 1, (0, 0, 0))
    surface.blit(actions_text, (hud_begin_x, actions_y))

    steps_y = 480
    steps_text = tiny_font.render('STEPS TAKEN: ' + str(steps_taken), 1, (0, 0, 0))
    surface.blit(steps_text, (hud_begin_x, steps_y))

    # CURRENT EPSILON

    epsilon_y = 530
    epsilon_text = tiny_font.render('CURRENT EPSILON: ' + str(current_epsilon), 1, (0, 0, 0))
    surface.blit(epsilon_text, (hud_begin_x, epsilon_y))

    # BEST EPOCH
    best_epoch_title_y = 600
    best_epoch_title_text = small_font.render('BEST EPOCH: ' + str(best_epoch), 1, (0, 0, 0))
    surface.blit(best_epoch_title_text, (
    hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - best_epoch_title_text.get_width() / 2,
    best_epoch_title_y + 25 - best_epoch_title_text.get_height() / 2))

    best_epoch_lines_y = 680
    best_epoch_lines_text = tiny_font.render('LINES: ' + str(best_lines), 1, (0, 0, 0))
    surface.blit(best_epoch_lines_text, (hud_begin_x, best_epoch_lines_y))

    best_epoch_score_y = 710
    best_epoch_score_text = tiny_font.render('SCORE: ' + str(best_score), 1, (0, 0, 0))
    surface.blit(best_epoch_score_text, (hud_begin_x, best_epoch_score_y))

    best_epoch_actions_y = 740
    best_epoch_actions_text = tiny_font.render('ACTIONS TAKEN: ' + str(best_actions), 1, (0, 0, 0))
    surface.blit(best_epoch_actions_text, (hud_begin_x, best_epoch_actions_y))

    best_epoch_steps_y = 770
    best_epoch_steps_text = tiny_font.render('STEPS TAKEN: ' + str(best_steps), 1, (0, 0, 0))
    surface.blit(best_epoch_steps_text, (hud_begin_x, best_epoch_steps_y))


# GAMEPLAY #

def generate_state(locked_positions, current_piece):
    """
    Computes the current state from the current game grid.

    The state will be store as a 20x10 numpy matrix, where each cell can have one of the following values:
    - 0: empty
    - 1: occupied (by a locked piece or the current piece)

    In order to not give the AI more information than what a human player would have, only the positions that can be
    viewed will be included into the state. This means that the four rows ABOVE the screen (where the piece spawns)
    are NOT included

    :param locked_positions: The current grid of the game
    :param current_piece: The current piece being played
    :return: The processed state
    """

    # Generate the initial grid (all 0s)
    grid = [[0 for _ in range(10)] for _ in range(20)]

    # For all positions in the locked grid, change the value to 1
    for (j, i) in locked_positions.keys():
        grid[i][j] = 1

    # Obtain the positions of the current piece and change them to 1
    piece_positions = generate_shape_positions(current_piece)
    for (x, y) in piece_positions:
        # Ignore negative positions (they're still out of bounds)
        if x >= 0 and y >= 0:
            grid[y][x] = 1

    return np.array(grid)


def generate_possible_actions(locked_positions, current_piece):
    """
    From the current locked positions and the current piece in place, generate all possible actions
    (all possible column and rotations for the current piece in play)

    This method is only used for the "new" approach (which considers the piece possible final positions as actions)
    Only legal final positions will be returned (positions that CAN be reached by the agent)

    The returned list of actions will have the structure
    [(x_column, rotation, state)]

    :param locked_positions: The current grid of the game
    :param current_piece: The current piece being played
    :return: A list A of possible actions
    """

    # Store the actions in a list
    actions = []

    # Store the initial piece X, Y and Rotation values
    original_x = current_piece.x
    original_y = current_piece.y
    original_rot = current_piece.rotation

    # Generate the current grid
    grid = create_grid(locked_positions)

    # Obtain all possible rotations for the current piece
    rotation_amount = len(current_piece.shape)

    # Loop, for all possible rotations
    for rot in range(rotation_amount):
        # For all possible X positions (from 0 to 9)
        for x in range(10):

            # Set the piece to the original position and rotation
            current_piece.x = original_x
            current_piece.y = original_y
            current_piece.rotation = original_rot

            # Boolean to check for legal moves
            legal_move = True

            # The piece will have to emulate being moved to the actual position
            # Rotations - Rotate the piece (lowering its depth with every rotation)
            for _ in range(rot):
                current_piece.rotation += 1
                current_piece.y += 1

                # If an illegal position is reached at any point, mark the action as illegal
                if not valid_space(current_piece, grid):
                    legal_move = False

            # Movements - Move the piece to the target position (lowering its depth with every movement)
            x_difference = current_piece.x - x

            # Difference < 0: move to the right / Difference > 0: move to the left
            if x_difference < 0:
                for _ in range(abs(x_difference)):
                    current_piece.x += 1
                    current_piece.y += 1

                    # If an illegal position is reached at any point, mark the action as illegal
                    if not valid_space(current_piece, grid):
                        legal_move = False
            elif x_difference > 0:
                for _ in range(x_difference):
                    current_piece.x -= 1
                    current_piece.y += 1

                    # If an illegal position is reached at any point, mark the action as illegal
                    if not valid_space(current_piece, grid):
                        legal_move = False

            # If the position was marked as illegal, this action is not possible: remove it
            if not legal_move:
                continue

            # While the position is valid, move down
            # We move down until the piece is placed down
            while valid_space(current_piece, grid):
                current_piece.y += 1

            # Once the position is not valid, move the piece upwards
            # (return to the last valid position)
            current_piece.y -= 1

            # Generate the current state and store it into the dictionary
            state = generate_state(locked_positions, current_piece)
            actions.append((x, rot, state))

    # Return the current piece to the original position
    current_piece.x = original_x
    current_piece.y = original_y
    current_piece.rotation = original_rot

    return actions


def generate_movement_sequence(action, current_piece):
    """
    From a specified action with shape (x, rotation, state), generates the movement sequence.

    The movement sequence is a queue of movements done by the agent to the piece.
    The moves always follow this order:
        1. Rotation
        2. Displacement (left or right)
        3. Hard drop (to place the action)

    This method is only used by agents of the new approximation (that have a different action definition)

    :param action: Action to be performed by the agent
    :param current_piece: Current piece to apply the action on
    :return: A queue of actions
    """

    # Queue to store the movements
    movements = deque()

    # Unpack the action
    action_x, rotation, _ = action

    # Add the rotations
    for i in range(rotation):
        movements.append('rotate')

    # Check the X difference
    x_difference = current_piece.x - action_x

    # If the difference is positive, the final position of the piece is to the LEFT of the current piece
    # Otherwise, it is to the RIGHT
    # If the difference is 0, there is no need to move the piece
    if x_difference < 0:
        while x_difference < 0:
            movements.append('right')
            x_difference += 1
    elif x_difference > 0:
        while x_difference > 0:
            movements.append('left')
            x_difference -= 1

    # Append a hard drop at the end, to lock the piece
    movements.append('hard_drop')

    return movements


def compute_aggregate_height(state):
    """
    Computes the aggregate height of a state (total sum of, for each column, the distance between the highest locked
    piece and the bottom)

    :param state: Current state (containing the piece)
    :return: Aggregate height
    """

    aggregate_height = 0

    # Obtain the state dimensions
    dimensions = state.shape

    # Loop by column, and then by row inside that column
    for x in range(dimensions[1]):
        for y in range(dimensions[0]):

            # Loop until a locked piece (or the current piece) has been found
            if state[y, x] != 0:
                aggregate_height += 20 - y
                break

    return aggregate_height


def compute_complete_lines(state):
    """
    Computes how many lines are fully complete (no holes)

    :param state: Current state (containing the piece)
    :return: Complete lines
    """

    # Compute which rows do not have holes
    full_rows = np.all(state != 0, axis=1)
    return np.sum(full_rows)


def compute_holes(state):
    """
    Computes the total number of holes in the game state

    :param state: Current state (containing the piece)
    :return: Number of holes
    """

    holes = 0

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


def compute_bumpiness(state):
    """
    Computes the bumpiness of the current game state (total sum of the absolute difference in height between contiguous
    columns)

    :param state: Current state (containing the piece)
    :return: Bumpiness value
    """

    bumpiness = 0

    # Get the dimensions of the state
    dimensions = state.shape

    # Store the height of the previous column
    previous_height = None

    # Loop by column, and then by row inside that column
    for x in range(dimensions[1]):

        # Store the current column height
        current_height = 0

        for y in range(dimensions[0]):

            # Loop until a locked piece (or the current piece) has been found
            if state[y, x] != 0:
                current_height = 20 - y
                break

        # Compare the height with the previous column
        if previous_height is None:
            # First column: cannot compare the height with the previous column
            previous_height = current_height
        else:
            # Next columns: obtain the absolute difference in height
            bumpiness += abs(current_height - previous_height)
            previous_height = current_height

    return bumpiness


def compute_heuristic_state_score(locked_pieces, current_piece, piece_locked):
    """
    From a game state, compute the heuristic score for the state

    The heuristic cost for the state is the following:
    score = -0.51 * (aggregate height) + 0.76 * (lines) - 0.36 * (holes) - 0.18 * (bumpiness)
    where the variables are, in order:
        - Total aggregate height of the board
        - Number of lines that would be cleared
        - Number of covered holes that would exist
        - Bumpiness (aggregate of height differences between columns)

    :param locked_pieces: Dictionary containing the currently locked pieces in the grid
    :param current_piece: Current piece in play
    :param piece_locked: TRUE if the current piece has been locked, FALSE otherwise

    :return: Score of the state
    """

    # Store the y position of the current piece
    current_piece_y = current_piece.y

    # Try to drop down the current piece if it is not already locked
    if not piece_locked:
        # Compute the grid
        grid = create_grid(locked_pieces)
        # Lower the piece until it touches the bottom
        while valid_space(current_piece, grid):
            current_piece.y += 1
        current_piece.y -= 1

    # Generate the state and return the piece to the correct location
    state = generate_state(locked_pieces, current_piece)
    current_piece.y = current_piece_y

    # Obtain the statistics
    aggregate_height = compute_aggregate_height(state)
    complete_lines = compute_complete_lines(state)
    holes = compute_holes(state)
    bumpiness = compute_bumpiness(state)

    # Return the score for the current state
    return -0.51 * aggregate_height + 0.76 * complete_lines - 0.36 * holes - 0.18 * bumpiness


def compute_reward_old(game_finished, piece_locked, lines_cleared, lowest_position_filled,
                       current_piece, locked_pieces, previous_score):
    """
    Computes the reward for a pair of state, action taking into account some details

    This method is only applied to "old" agents

    There are currently two possible approaches to the rewards method:
    * 'game': Game score based approach. Gives scores to the action depending on how has the action affected the score
    * 'heuristic': Heuristic based approach. Scores both the original state and the state after applying the action
                   using a heuristic function.
                   The reward is (new state heuristic) - (previous state heuristic).

    :param game_finished: TRUE if the action caused the end of the game, FALSE otherwise
    :param piece_locked: TRUE if the piece has been locked, FALSE otherwise
    :param lines_cleared: (Only if piece_locked = TRUE) How many lines were cleared with the locked piece.
    :param lowest_position_filled: (Only if piece_locked = TRUE) The lowest Y value reached by the locked piece.
    :param current_piece: Current position of the piece
    :param locked_pieces: Currently locked pieces in the grid
    :param previous_score: Score of the previous state
    :return: (reward, score)
             WHERE
             reward: The reward for the pair state, action
             score: The score awarded to the current state (heuristic approach) or None (game approach)
    """

    # Reward computed depends on the method:

    if rewards_method == 'game':
        """
        Game reward
        
        The reward computed is as follows:
        * If the game is finished, REWARD = -2 (we don't want the agent to lose)
        * If no piece has been locked: REWARD = -0.1 (we want the agent to act)
        * If a piece has been locked:
            * No line cleared, REWARD = (lowest_position_filled / 10) - 1 (locking pieces is good, the deeper the better)
              However, locking pieces above half of the stack height starts being penalized.
            * Lines cleared, REWARD = 2^lines_cleared (the more lines that are cleared at once, the better the state is)
        """

        # Game finished: immediate big penalty
        if game_finished:
            return -2, None
        # Game not finished but piece not locked: very small penalty (want the agent to try to lock fast)
        elif not piece_locked:
            return -0.1, None
        # Piece locked: compute a reward appropiately
        else:
            # 0 lines locked: reward based on the lowest depth filled
            if lines_cleared == 0:
                return (lowest_position_filled / 10) - 1, None
            # 1 or more lines locked: give the agent a bigger reward (more lines = better reward)
            else:
                return 2 ** lines_cleared, None
    else:
        """
        Heuristic reward
        
        The game will take into account the following values (after the action has already applied, and assuming that
        the piece would instantly fall down at the current position with the current rotation):
        - Total aggregate height of the board
        - Number of lines that would be cleared
        - Number of covered holes that would exist
        - Bumpiness (aggregate of height differences between columns)
        
        A state will be scored with the following formula:
        score = -0.51 * (aggregate height) + 0.76 * (lines) - 0.36 * (holes) - 0.18 * (bumpiness)
        
        The reward will be:
        * If there is a game over, REWARD = -2 (harsh penalty for losing the game)
        * If no lines were cleared, REWARD = (current score) - (previous score) (the difference in scores)
        * If a line was cleared, REWARD = 2^lines_cleared (big reward for line clears)
        """

        # Check for cases where computing the reward is not needed
        # It's not a problem that we don't return a state score in these cases. In both cases, the score would be
        # irrelevant (either the game has restarted or we are working with a new piece that will need its own rewards)

        if game_finished:
            # Game finished: return a big negative reward
            return -2, None
        elif lines_cleared > 0:
            # Lines have been cleared: return a positive reward depending on the amount of cleared lines
            return 2 ** lines_cleared, None
        else:
            # Compute the score
            current_state_score = compute_heuristic_state_score(locked_pieces, current_piece, piece_locked)

            # Compute the reward
            reward = current_state_score - previous_score

            # If the piece was locked, return a null score instead
            # (Scores are reset for every piece)
            if piece_locked:
                current_state_score = None

            return reward, current_state_score


def compute_reward_new(game_finished, lines_cleared, original_score, new_score, lowest_position_filled):
    """
    Computes the reward for a pair of state, action (where action is contained within reached_state)

    This method is only used for "new" agents

    Both a heuristic and game approach are used for the rewards. However, the game approach has been modified
    to take into account the new actions.

    :param game_finished: TRUE if the action made the game end, FALSE otherwise
    :param lines_cleared: Amount of lines that have been cleared by the action
    :param original_score: Score given to the original state
    :param new_score: Score given to the state reached after applying the action
    :param lowest_position_filled: The lowest position filled by the piece (on the Y axis)

    :return: The reward given to the current state
    """

    if rewards_method == "game":
        """
        Game reward
        
        In this case, the reward is based directly on the score obtained by the agent by placing pieces,
        but slightly modified to better suit the rewards
        
        To be precise, the reward will be:
        * If there is a game over, REWARD = -1 (penalty for losing the game)
        * If no lines were cleared, REWARD = (lowest_position_filled / 10) - 1 (from +1 at the lowest depth 
                                                                                to -1 at the highest depth)
        * If a line was cleared, REWARD = 2^lines_cleared (big reward for line clears)
        """
        if game_finished:
            # Game finished: return a big negative reward
            return -2
        elif lines_cleared > 0:
            # Lines have been cleared: return a positive reward depending on the amount of cleared lines
            return 2 ** lines_cleared
        else:
            # No lines have been cleared: small reward (placing the piece lower is more valuable)
            return (lowest_position_filled / 10) - 1

    else:
        """
        Heuristic reward
    
        The game will take into account the following values (after the action has already applied, and assuming that
        the piece would instantly fall down at the current position with the current rotation):
        - Total aggregate height of the board
        - Number of lines that would be cleared
        - Number of covered holes that would exist
        - Bumpiness (aggregate of height differences between columns)
    
        A state will be scored with the following formula:
        score = -0.51 * (aggregate height) + 0.76 * (lines) - 0.36 * (holes) - 0.18 * (bumpiness)
    
        The reward will be:
        * If there is a game over, REWARD = -2 (harsh penalty for losing the game)
        * If no lines were cleared, REWARD = (current score) - (previous score) (the difference in scores)
        * If a line was cleared, REWARD = 2^lines_cleared (big reward for line clears)
        """

        if game_finished:
            # Game finished: return a big negative reward
            return -2
        elif lines_cleared > 0:
            # Lines have been cleared: return a positive reward depending on the amount of cleared lines
            return 2 ** lines_cleared
        else:
            # No lines have been cleared: the reward is the difference in evaluation between the new and the previous score
            return new_score - original_score


###############################
# MAIN LOOP AUXILIARY METHODS #
###############################

# These methods are the main loop methods shared between all instances of the main game loop
# (human player, AI player or AI learner)

def initialize_game():
    """
    Initializes all necessary variables for the main game loop, and returns them

    This has been taken into a separate method since it is shared by all three loops

    :returns tuple(locked_positions, current_speed, score, lines, level, change_piece, run, randomizer_shapes,
            current_piece, next_piece, clock, fall_time)
            (meaning of each value explained in the code)
    """

    # Variables used by the main loop #

    # Grid with all the locked positions
    locked_positions = {}

    # Speed at which the pieces fall (to be updated during the loop)
    # Speed can be increased up to 9 times at most
    current_speed = initial_speed

    # Score, lines cleared and level reached
    score = 0
    lines = 0
    level = 0
    # Piece is locked, need a new piece
    change_piece = False
    # Game is still running
    run = True

    # Generates an initial list of shapes
    randomizer_shapes = bag_randomizer()

    # Get the initial piece and the initial next piece
    current_piece, randomizer_shapes = get_shape(randomizer_shapes)
    next_piece, randomizer_shapes = get_shape(randomizer_shapes)

    # Initializes the clock and the time counters
    # (The human player uses a real-time clock)
    clock = pygame.time.Clock()
    fall_time = 0

    # Start playing the song (if sound is active)
    play_song()

    # Return all initialized variables
    return (locked_positions, current_speed, score, lines, level, change_piece, run, randomizer_shapes,
            current_piece, next_piece, clock, fall_time)


def process_inputs(inputs, current_piece, grid):
    """
    Given a list of inputs, processes them and applies them to the board

    Note that inputs (even if there is only one input) must be passed as a list.
    Also note that the ESC input must be processed outside of this method (since it breaks the loop)

    :param inputs: List containing all inputs to be processed
    :param current_piece: Piece currently in play
    :param grid: Current grid of the game
    :return: Updated current_piece, grid and change_piece
    """

    # Sets the change_piece to False (change_piece computes if the piece should be locked or not,
    # only the hard drop action instantly locks a piece)
    change_piece = False

    # Loop through the list of actions
    for action in inputs:

        # Left (move left and play the appropriate sound)
        if action == "left":
            current_piece.x -= 1
            if not valid_space(current_piece, grid):
                current_piece.x += 1
            play_sound("action")

        # Right (move right and play the appropriate sound)
        if action == "right":
            current_piece.x += 1
            if not valid_space(current_piece, grid):
                current_piece.x -= 1
            play_sound("action")

        # Soft drop (moves the piece down a position)
        if action == "soft_drop":
            current_piece.y += 1
            if not valid_space(current_piece, grid):
                current_piece.y -= 1

        # Hard / instant drop (instantly moves the piece to the lowest position it can move)
        if action == "hard_drop":
            # Try to move the piece down until an illegal position is reached, and then move upwards to reach
            # the final valid position
            while valid_space(current_piece, grid):
                current_piece.y += 1
            current_piece.y -= 1

            # After a hard drop, piece will be guaranteed to be locked
            change_piece = True

        # Rotation (rotates the piece into the next rotation)
        if action == "rotate":
            current_piece.rotation += 1
            if not valid_space(current_piece, grid):
                current_piece.rotation -= 1
            play_sound("action")

    # Returns all values
    return current_piece, grid, change_piece


def place_piece(current_piece, grid):
    """
    Inserts the current piece into the grid

    :param current_piece: Piece currently in play
    :param grid: Current state of the grid
    :return: Updated grid and shape_pos (current_piece converted into grid positions)
    """

    shape_pos = generate_shape_positions(current_piece)
    for i in range(len(shape_pos)):
        x, y = shape_pos[i]
        if y > -1:
            grid[y][x] = current_piece.color

    return grid, shape_pos


#####################
# MAIN LOOP METHODS #
#####################

def main_human_player(win):
    """
    Main logic of the game when a human player is active

    AI players and training modes have their own separate logic, adapted to the specific necessities

    :param win: Surface used to draw all the elements.
    """

    # Initialize all necessary variables
    (locked_positions, current_speed, score, lines, level, change_piece, run, randomizer_shapes, current_piece,
     next_piece, clock, fall_time) = initialize_game()

    # While the game is not over (main logic loop)
    while run:

        # Create the grid and update all the clocks, marking a new tick
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        # Create a list to contain all processed inputs
        actions = []

        # Process all the player inputs
        for event in pygame.event.get():

            # Window has been closed
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

            # Key has been pressed
            if event.type == pygame.KEYDOWN:

                # ESC key (exit to main menu)
                if event.key == pygame.K_ESCAPE:
                    run = False
                    stop_sounds()

                # Left key (left action)
                if event.key == pygame.K_LEFT:
                    actions.append("left")

                # Right key (right action)
                if event.key == pygame.K_RIGHT:
                    actions.append("right")

                # Down key (soft drop action)
                if event.key == pygame.K_DOWN:
                    actions.append("soft_drop")

                # Up key (hard/instant drop action)
                if event.key == pygame.K_UP:
                    actions.append("hard_drop")

                # R key (rotation and play the appropriate sound)
                if event.key == pygame.K_r:
                    actions.append("rotate")

        # Execute all processed inputs
        current_piece, grid, change_piece = process_inputs(actions, current_piece, grid)

        # Clock calculations (make the piece fall)
        if fall_time > current_speed:
            fall_time = 0
            current_piece.y += 1
            # If, after lowering the piece, it reaches an invalid position, it has touched another piece: lock it
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        # Place the current piece into the grid
        grid, shape_pos = place_piece(current_piece, grid)

        # If the piece has been locked in place
        if change_piece:

            # Add the current piece to locked positions (conserving the biggest y value)
            # The biggest Y value is stored to compute the score later
            shape_y = -1
            for pos in shape_pos:
                if pos[1] > shape_y:
                    shape_y = pos[1]
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color

            # Get the next piece
            current_piece = next_piece
            next_piece, randomizer_shapes = get_shape(randomizer_shapes)

            # Update lines cleared
            lines_cleared = clear_rows(grid, locked_positions)
            lines += len(lines_cleared)

            # Compute the new score
            score_increase = compute_score(len(lines_cleared), shape_y, level)

            # Update the level and speed
            level = lines // 10
            current_speed = initial_speed - speed_modifier * level
            # Ensure that the speed doesn't go below a limit
            if current_speed < minimum_speed:
                current_speed = minimum_speed

            # Play the piece lock sound
            play_sound("fall")

            # Check if an animation needs to be played (lines have been cleared)
            if len(lines_cleared) > 0:

                # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                play_sound("line")

                # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                draw_clear_row(win, lines_cleared)

            # Update the score
            score += score_increase

            # Tick the clock again, to ensure that no time is lost due to the processing
            clock.tick()

        # Draw everything and update the screen
        draw_manager(win, grid, current_piece, next_piece, score, level, lines)
        pygame.display.flip()

        # Check if the game has ended
        if check_defeat(locked_positions):
            stop_sounds()
            play_sound("lost")
            draw_game_over_effect(win)
            run = False

# ORIGINAL APPROACH ("OLD") #
# This approach considers an action as a player input


def main_ai_player_old(win):
    """
    Main logic of the game when a AI player (using an "old" approach agent) is active

    The main differences are that the actions are now polled from the agent instead of from the user inputs
    Once the game is finished, the score will be displayed on the console

    :param win: Surface used to draw all the elements.
    """

    # Initialize all necessary variables
    (locked_positions, _, score, lines, level, change_piece, run, randomizer_shapes, current_piece,
     next_piece, clock, fall_time) = initialize_game()

    # Current speed will be fixed to the specified value (and not updated)
    current_speed = game_speed_ai

    # AI: Poll time is used to check for the AI player actions
    poll_time = 0

    # AI: Keep Action and Q-Values outside of the loop (to keep their values)
    # They are initialized with placeholder values
    action = "No actions taken yet"
    q_values = np.array([[0, 0, 0, 0]])

    # While the game is not over (main logic loop)
    while run:

        # Create the grid and update all the clocks, marking a new tick
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        poll_time += clock.get_rawtime()
        clock.tick()

        # Unless specified otherwise, the piece is not going to be locked
        change_piece = False

        # Check if the player has exited the game or has pressed the ESC key
        for event in pygame.event.get():

            # Window has been closed
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

            # Key has been pressed
            if event.type == pygame.KEYDOWN:

                # ESC key (exit to main menu)
                if event.key == pygame.K_ESCAPE:
                    run = False
                    stop_sounds()

        # Prepare the current state for the AI
        current_state = generate_state(locked_positions, current_piece)

        # Clock calculations
        # The order of these calculations is relevant. The movement must be polled first always
        # This ensures it remains consistent with the human behaviour (movement first, locking second)

        # 1 - If needed, poll the agent for an action and execute it
        if poll_time > polling_speed:
            poll_time = 0
            action, q_values = agent.act(current_state)
            current_piece, grid, change_piece = process_inputs([action], current_piece, grid)

        # 2 - If needed, move the piece downwards
        if fall_time > current_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        # Place the current piece into the grid
        grid, shape_pos = place_piece(current_piece, grid)

        # If the piece has been locked in place
        if change_piece:
            # Add the current piece to locked positions (conserving the biggest y value)
            shape_y = -1
            for pos in shape_pos:
                if pos[1] > shape_y:
                    shape_y = pos[1]
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color

            # Get the next piece
            current_piece = next_piece
            next_piece, randomizer_shapes = get_shape(randomizer_shapes)

            # Update lines
            lines_cleared = clear_rows(grid, locked_positions)
            lines += len(lines_cleared)

            # Compute the score increase
            score_increase = compute_score(len(lines_cleared), shape_y, level)

            # Update level (speed is not increased in AI mode)
            level = lines // 10

            # Play the piece lock sound
            play_sound("fall")

            # Check if a line animation has to be played
            if len(lines_cleared) > 0:
                # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                play_sound("line")

                # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                draw_ai_player_old_information(win, current_state, q_values, action, agent.actions_performed)
                draw_clear_row(win, lines_cleared)

            # Update the score
            score += score_increase

            # Prepare everything for the next loop
            clock.tick()

        # Draw everything (original HUD and AI HUD)
        draw_manager(win, grid, current_piece, next_piece, score, level, lines)
        draw_ai_player_old_information(win, current_state, q_values, action, agent.actions_performed)
        # Update the screen
        pygame.display.flip()

        # Check if the game has ended
        if check_defeat(locked_positions):
            stop_sounds()
            play_sound("lost")
            draw_game_over_effect(win)
            run = False

        # If the agent has cleared more than the specified amount, cut it short
        if lines >= max_lines_training:
            run = False

    # Game has ended: print results
    print("END RESULTS: ")
    print("Lines: " + str(lines))
    print("Score: " + str(score))
    print("Actions taken: " + str(agent.actions_performed))


def main_ai_learn_old(win):
    """
    Main logic of the game when the AI (using an "old" mode agent) is in learning mode

    This method has several differences compared to the standard AI player loop:
    - The game automatically replays itself after every epoch
    - The game can run either on a real clock or a logical clock (when running in fast mode)

    :param win: Surface used to draw all the elements.
    """

    # Initialize the epoch counter
    current_epoch = 1

    # Store info about the best epoch (to display it)
    best_epoch = -1
    best_score = 0
    best_lines = 0
    best_actions = 0

    # Game is played until the max epoch is reached
    while current_epoch <= total_epochs:

        # GAME START:

        # Initialize all necessary variables
        (locked_positions, _, score, lines, level, change_piece, run, randomizer_shapes, current_piece,
         next_piece, clock, fall_time) = initialize_game()

        # Piece fall speed will not change during the game, so it will not be modified
        current_speed = game_speed_ai

        # If the game is in fast mode, the clock itself will actually be ignored
        # AI: Poll time is used to check for the AI player actions
        poll_time = 0

        # Information that needs to be stored outside of the loop to compute rewards
        # (only used when using a heuristic approach)
        previous_state_score = None

        # Information that needs to be stored outside for the extra HUD information (starting with some default values)
        hud_current_state = [[0 for _ in range(10)] for _ in range(20)]
        hud_next_state = hud_current_state
        hud_action = "placeholder"
        hud_reward = 0.0

        # While the game is not over (main logic loop)
        while run:

            # Create the grid
            grid = create_grid(locked_positions)

            # Check if fast mode is active
            # NOT ACTIVE: The real-time clock is used
            if not fast_training:
                fall_time += clock.get_rawtime()
                poll_time += clock.get_rawtime()
                clock.tick()
            # ACTIVE: automatically increase the counters by the polling speed
            else:
                fall_time += polling_speed
                poll_time += polling_speed

            # The piece is not going to change (unless otherwise specified)
            change_piece = False

            # Check if the player has exited the game
            # Train mode cannot be exited using ESC (needs to be canceled through the console or closing the window)
            for event in pygame.event.get():

                # Window has been closed
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()

            # Prepare the current state for the AI
            current_state = generate_state(locked_positions, current_piece)

            # If no previous state heuristic cost is present (new game or a piece was just locked), compute it
            if previous_state_score is None:
                previous_state_score = compute_heuristic_state_score(locked_positions, current_piece, change_piece)

            # Clock calculations
            # The order of these calculations is relevant. The movement must be polled first always
            # This ensures it remains consistent with the human behaviour (movement first, locking second)

            # Store the action performed (used to store the experience)
            action = None

            # 1 - If needed, poll the agent for an action and execute it
            if poll_time > polling_speed:
                poll_time = 0
                # Get the action and the q-values
                action, q_values = agent.act(current_state)
                # Act
                current_piece, grid, change_piece = process_inputs([action], current_piece, grid)

            # 2 - If needed, move the piece downwards
            if fall_time > current_speed:
                fall_time = 0
                current_piece.y += 1
                if not valid_space(current_piece, grid) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            # Place the current piece into the grid
            grid, shape_pos = place_piece(current_piece, grid)

            # Variables used later to store the experience #

            # Remember if the piece has been locked or not (since the value is changed after processing the piece lock)
            # This is needed to store the experience
            final_state = change_piece

            # Store the lowest Y value (used to compute the reward)
            lowest_y = -1

            # Store the number of lines cleared
            lines_cleared_store = 0

            ###

            # If the piece has been locked in place
            if change_piece:
                # Add the current piece to locked positions (conserving the biggest y value)
                shape_y = -1
                for pos in shape_pos:
                    if pos[1] > shape_y:
                        shape_y = pos[1]
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color

                # Update the lowest Y outside
                lowest_y = shape_y

                # Get the next piece
                current_piece = next_piece
                next_piece, randomizer_shapes = get_shape(randomizer_shapes)

                # Update lines
                lines_cleared = clear_rows(grid, locked_positions)
                lines += len(lines_cleared)

                # Compute the score increase
                score_increase = compute_score(len(lines_cleared), shape_y, level)

                # Update the level (speed does not increase in AI mode)
                level = lines // 10

                # Update the lines cleared outside
                lines_cleared_store = len(lines_cleared)

                # Play the piece lock sound
                play_sound("fall")

                # Check if a line clear needs to be animated
                if len(lines_cleared) > 0:
                    # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                    play_sound("line")

                    # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                    if not fast_training:
                        draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                        draw_ai_learn_old_information(win,
                                                      hud_current_state,
                                                      hud_next_state,
                                                      hud_action,
                                                      hud_reward,
                                                      current_epoch,
                                                      agent.actions_performed,
                                                      agent.epsilon,
                                                      best_epoch,
                                                      best_score,
                                                      best_lines,
                                                      best_actions)
                        draw_clear_row(win, lines_cleared)

                # Update the score
                score += score_increase

                # Only tick if not in fast mode
                if not fast_training:
                    clock.tick()

            # Check if the game has ended
            if check_defeat(locked_positions):
                run = False

            # If an action has been taken, prepare everything to store the experience
            if action is not None:
                # State reached after the action
                new_state = generate_state(locked_positions, current_piece)
                # Reward for the state/action pair
                reward, previous_state_score = compute_reward_old(not run,
                                                                  final_state,
                                                                  lines_cleared_store,
                                                                  lowest_y,
                                                                  current_piece,
                                                                  locked_positions,
                                                                  previous_state_score)

                # Store the experience into the agent
                agent.insert_experience(current_state, action, reward, new_state, final_state)

                # Check if the best epoch needs to be updated
                if lines_cleared_store > best_lines or (lines_cleared_store == best_lines and score > best_score) or (lines_cleared_store == best_lines and score == best_score and agent.actions_performed > best_actions):
                    best_epoch = current_epoch
                    best_lines = lines_cleared_store
                    best_score = score
                    best_actions = agent.actions_performed

                # Update the stored info for the hud
                hud_current_state = current_state
                hud_next_state = new_state
                hud_action = action
                hud_reward = reward

            # Draw everything (original HUD and AI HUD) IF not in fast mode
            if not fast_training:
                draw_manager(win, grid, current_piece, next_piece, score, level, lines)
                draw_ai_learn_old_information(win,
                                              hud_current_state,
                                              hud_next_state,
                                              hud_action,
                                              hud_reward,
                                              current_epoch,
                                              agent.actions_performed,
                                              agent.epsilon,
                                              best_epoch,
                                              best_score,
                                              best_lines,
                                              best_actions)
                # Update the screen
                pygame.display.flip()

            # If the agent has cleared more than the specified amount, cut it short
            if lines >= max_lines_training:
                run = False

        # GAME END

        # Advance to the next epoch
        current_epoch += 1

        # Notify the agent of the new epoch
        agent.finish_epoch(lines, score)


# SECOND APPROACH ("NEW") #
# This approach considers all posible final positions of a piece as actions

def main_ai_player_new(win):
    """
    Main logic of the game when a AI player (using a "new" approach agent) is active

    In addition to the main game loop, an extra pre-loop is added.
    In this pre-loop, the action and sequence of steps performed by the agent is computed.
    This will later be used during the main game loop itself.

    :param win: Surface used to draw all the elements.
    """

    # Initialize all necessary variables
    (locked_positions, _, score, lines, level, change_piece, run, randomizer_shapes, current_piece,
     next_piece, clock, fall_time) = initialize_game()

    # Current speed will be fixed to the specified value (and not updated)
    current_speed = game_speed_ai

    # AI: Poll time is used to check for the AI player actions
    poll_time = 0

    # HUD-RELATED - Store the last "step" taken while in the loop
    # Outside of the run loop to avoid the value being crushed while doing a new action
    last_step = None

    # While the game is not over (main logic loop)
    while run:

        # PRE-LOOP
        # Generate all possible actions for the current state and pieces
        possible_actions = generate_possible_actions(locked_positions, current_piece)

        # Choose an action from the agent (and store the Q-Value)
        action, q_value = agent.act(possible_actions)

        # Generate the steps to be taken while in-game
        steps = generate_movement_sequence(action, current_piece)

        # Bool used to keep track of when the game loop should yield (to compute the next action)
        loop_ended = False

        # Tick the clock (to "ignore" time lost during computations)
        clock.tick()

        # GAME LOOP
        # The game loop will run until either the game is over, all steps have been taken or
        # the piece has been locked inside. This will be controled by a variable outside
        while not loop_ended:

            # Create the grid and update all the clocks, marking a new tick
            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            poll_time += clock.get_rawtime()
            clock.tick()

            # Unless specified otherwise, the piece is not going to be locked
            change_piece = False

            # Check if the player has exited the game or has pressed the ESC key
            for event in pygame.event.get():

                # Window has been closed
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()

                # Key has been pressed
                if event.type == pygame.KEYDOWN:

                    # ESC key (exit to main menu)
                    if event.key == pygame.K_ESCAPE:
                        run = False
                        stop_sounds()

            # Clock calculations
            # The order of these calculations is relevant. The movement must always be polled first
            # This ensures it remains consistent with the human behaviour (movement first, locking second)

            # 1 - If needed, execute an action
            if poll_time > polling_speed:
                step = steps.popleft()
                last_step = step
                poll_time = 0
                current_piece, grid, change_piece = process_inputs([step], current_piece, grid)

                # Notify the agent
                agent.notify_step()

            # 2 - If needed, move the piece downwards
            if fall_time > current_speed:
                fall_time = 0
                current_piece.y += 1
                if not valid_space(current_piece, grid) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            # Place the current piece into the grid
            grid, shape_pos = place_piece(current_piece, grid)

            # If the piece has been locked in place
            if change_piece:
                # Add the current piece to locked positions (conserving the biggest y value)
                shape_y = -1
                for pos in shape_pos:
                    if pos[1] > shape_y:
                        shape_y = pos[1]
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color

                # Get the next piece
                current_piece = next_piece
                next_piece, randomizer_shapes = get_shape(randomizer_shapes)

                # Update lines
                lines_cleared = clear_rows(grid, locked_positions)
                lines += len(lines_cleared)

                # Compute the score increase
                score_increase = compute_score(len(lines_cleared), shape_y, level)

                # Update level (speed is not increased in AI mode)
                level = lines // 10

                # Play the piece lock sound
                play_sound("fall")

                # Check if a line animation has to be played
                if len(lines_cleared) > 0:
                    # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                    play_sound("line")

                    # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                    draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                    draw_ai_player_new_information(win,
                                                   action[2],
                                                   q_value,
                                                   last_step,
                                                   agent.actions_performed,
                                                   agent.displacements)
                    draw_clear_row(win, lines_cleared)

                # Update the score
                score += score_increase

                # End the loop (to continue with either the pre-loop or finish the game)
                loop_ended = True

                # Prepare everything for the next loop
                clock.tick()

            # Draw everything (original HUD and AI HUD)
            draw_manager(win, grid, current_piece, next_piece, score, level, lines)
            draw_ai_player_new_information(win,
                                           action[2],
                                           q_value,
                                           last_step,
                                           agent.actions_performed,
                                           agent.displacements)
            # Update the screen
            pygame.display.flip()

            # Check if the game has ended
            if check_defeat(locked_positions):
                stop_sounds()
                play_sound("lost")
                draw_game_over_effect(win)
                run = False
                loop_ended = True

            # If the agent has cleared more than the specified amount, cut it short
            if lines >= max_lines_training:
                run = False
                loop_ended = True

            # Check if the queue is empty
            if len(steps) <= 0:
                loop_ended = True

    # Game has ended: print results
    print("END RESULTS: ")
    print("Lines: " + str(lines))
    print("Score: " + str(score))
    print("Actions taken: " + str(agent.actions_performed))


def main_ai_learn_new(win):
    """
        Main logic of the game when the AI (using a "new" approach agent) is in learning mode

        This method has several differences compared to the standard AI player loop:
            - The game automatically replays itself after every epoch
            - The game can run either on a real clock or a logical clock (when running in fast mode)

        :param win: Surface used to draw all the elements.
        """

    # Initialize the epoch counter
    current_epoch = 1

    # Store info about the best epoch (used to display it)
    best_epoch = -1
    best_score = 0
    best_lines = 0
    best_actions = 0
    best_steps = 0

    # Game is played until the max epoch is readched
    while current_epoch <= total_epochs:

        # GAME START:

        # Initialize all necessary variables
        (locked_positions, _, score, lines, level, change_piece, run, randomizer_shapes, current_piece,
         next_piece, clock, fall_time) = initialize_game()

        # Current speed will be fixed to the specified value (and not updated)
        current_speed = game_speed_ai

        # AI: Poll time is used to check for the AI player actions
        poll_time = 0

        # While the game is not over (main logic loop)
        while run:

            # PRE-LOOP
            # Generate all possible actions for the current state and pieces
            possible_actions = generate_possible_actions(locked_positions, current_piece)

            # Choose an action from the agent (and store the Q-Value)
            action, q_value = agent.act(possible_actions)

            # Generate the steps to be taken while in-game
            steps = generate_movement_sequence(action, current_piece)

            # Bool used to keep track of when the game loop should yield (to compute the next action)
            loop_ended = False

            # TRAINING RELATED VARIABLES:
            # Compute the initial state and score
            initial_state = generate_state(locked_positions, current_piece)
            initial_score = compute_heuristic_state_score(locked_positions, current_piece, False)

            # Store the final state and the reward obtained
            final_state = action[2]
            final_score = None

            # Store the last amount of lines cleared
            final_lines_cleared = 0

            # Store the depth at which the piece was locked
            final_piece_depth = 0

            # Tick the clock (to "ignore" time lost during computations)
            clock.tick()

            # GAME LOOP
            # The game loop will run until either the game is over, all steps have been taken or
            # the piece has been locked inside. This will be controled by a variable outside
            while not loop_ended:

                # Create the grid
                grid = create_grid(locked_positions)

                # Check if fast mode is active
                # NOT ACTIVE: The real-time clock is used
                if not fast_training:
                    fall_time += clock.get_rawtime()
                    poll_time += clock.get_rawtime()
                    clock.tick()
                # ACTIVE: Automatically increase the counters by the polling speed
                else:
                    fall_time += polling_speed
                    poll_time += polling_speed

                # Unless specified otherwise, the piece is not going to be locked
                change_piece = False

                # Check if the player has exited the game or has pressed the ESC key
                for event in pygame.event.get():

                    # Window has been closed
                    if event.type == pygame.QUIT:
                        pygame.display.quit()
                        pygame.quit()
                        sys.exit()

                # Clock calculations
                # The order of these calculations is relevant. The movement must always be polled first
                # This ensures it remains consistent with the human behaviour (movement first, locking second)

                # 1 - If needed, execute an action
                if poll_time > polling_speed:
                    poll_time = 0
                    current_piece, grid, change_piece = process_inputs([steps.popleft()], current_piece, grid)

                    # Notify the agent of the step
                    agent.notify_step()

                # 2 - If needed, move the piece downwards
                if fall_time > current_speed:
                    fall_time = 0
                    current_piece.y += 1
                    if not valid_space(current_piece, grid) and current_piece.y > 0:
                        current_piece.y -= 1
                        change_piece = True

                # Place the current piece into the grid
                grid, shape_pos = place_piece(current_piece, grid)

                # If the piece has been locked in place
                if change_piece:
                    # Add the current piece to locked positions (conserving the biggest y value)
                    shape_y = -1
                    for pos in shape_pos:
                        if pos[1] > shape_y:
                            shape_y = pos[1]
                        p = (pos[0], pos[1])
                        locked_positions[p] = current_piece.color

                    # Store the depth
                    final_piece_depth = shape_y

                    # Compute the final state score and store it
                    final_score = compute_heuristic_state_score(locked_positions, current_piece, True)

                    # Get the next piece
                    current_piece = next_piece
                    next_piece, randomizer_shapes = get_shape(randomizer_shapes)

                    # Update lines
                    lines_cleared = clear_rows(grid, locked_positions)
                    lines += len(lines_cleared)

                    # Store the amount of cleared lines
                    final_lines_cleared = len(lines_cleared)

                    # Compute the score increase
                    score_increase = compute_score(len(lines_cleared), shape_y, level)

                    # Update level (speed is not increased in AI mode)
                    level = lines // 10

                    # Play the piece lock sound
                    play_sound("fall")

                    # Check if a line animation has to be played
                    if len(lines_cleared) > 0:
                        # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                        play_sound("line")

                        # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                        if not fast_training:
                            draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                            draw_ai_learn_new_information(win,
                                                          current_epoch,
                                                          initial_state,
                                                          final_state,
                                                          q_value,
                                                          agent.actions_performed,
                                                          agent.displacements,
                                                          agent.epsilon,
                                                          best_epoch,
                                                          best_lines,
                                                          best_score,
                                                          best_actions,
                                                          best_steps)
                            draw_clear_row(win, lines_cleared)

                    # Update the score
                    score += score_increase

                    # End the loop (to continue with either the pre-loop or finish the game)
                    loop_ended = True

                    # Prepare everything for the next loop
                    clock.tick()

                # Draw everything (original HUD and AI HUD)
                if not fast_training:
                    draw_manager(win, grid, current_piece, next_piece, score, level, lines)
                    draw_ai_learn_new_information(win,
                                                  current_epoch,
                                                  initial_state,
                                                  final_state,
                                                  q_value,
                                                  agent.actions_performed,
                                                  agent.displacements,
                                                  agent.epsilon,
                                                  best_epoch,
                                                  best_lines,
                                                  best_score,
                                                  best_actions,
                                                  best_steps)
                    # Update the screen
                    pygame.display.flip()

                # Check if the game has ended
                if check_defeat(locked_positions):
                    run = False
                    loop_ended = True

                # If the agent has cleared more than the specified amount, cut it short
                if lines >= max_lines_training:
                    run = False
                    loop_ended = True

                # Check if the queue is empty
                if len(steps) <= 0:
                    loop_ended = True

            # POST-LOOP

            # Obtain the reward
            reward = compute_reward_new(not run,
                                        final_lines_cleared,
                                        initial_score,
                                        final_score,
                                        final_piece_depth)

            # Store the experience
            agent.insert_experience(initial_state, reward, final_state, not run)

            # Check if the best epoch needs to be updated
            if lines > best_lines or (lines == best_lines and score > best_score) or (
                    lines == best_lines and score == best_score and agent.actions_performed > best_actions):
                best_epoch = current_epoch
                best_lines = lines
                best_score = score
                best_actions = agent.actions_performed
                best_steps = agent.displacements

        # GAME END

        # Advance to the next epoch
        current_epoch += 1

        # Notify the agent of the new epoch
        agent.finish_epoch(lines, score)


def menu_logic(win):
    """
    Function that draws the main menu of the game.
    Depending on the type of player (human or AI), a different game logic will be launched

    :param win: Surface to draw everything on.
    """

    run = True

    # While the game is not closed
    while run:
        # Draw the main menu
        draw_main_menu(win)

        # Check for player inputs
        for event in pygame.event.get():
            # If the window is closed, simply exit the loop
            if event.type == pygame.QUIT:
                run = False
            # If a key is pressed
            if event.type == pygame.KEYDOWN:
                # If escape is pressed, close the game
                if event.key == pygame.K_ESCAPE:
                    run = False
                # Else, start the appropriate game logic depending on the type of player and the type of agent
                else:
                    if not ai_player:
                        main_human_player(win)
                        pygame.event.clear()
                    else:
                        if agent.agent_type == "old":
                            main_ai_player_old(win)
                        else:
                            main_ai_player_new(win)
                        pygame.event.clear()

    # When closed, exit everything in an ordered way
    pygame.display.quit()
    pygame.quit()
    sys.exit()


# Code to be executed if called directly
if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Starts a game of TETRIS for a human player with sound."
                                                 "\nThe game can be customized by indicating arguments")

    ####################################
    # ARGUMENTS - PLAYER-SET VARIABLES #
    ####################################

    # NOTE: All arguments are optional
    # By default, launching the script without arguments will start the game for a human player with sound.

    # SILENT (-sil or --silent) - If the argument is present, the game will be soundless
    parser.add_argument('-sil',
                        '--silent',
                        action='store_true',
                        help="Disables sound for the game")

    # FIXED SPEED (-fs or --fixedspeed) - (HUMAN ONLY) Sets the speed to a fixed speed (speed will not increase with
    # difficulty). This setting emulates the simplified speed setting used for AI players
    parser.add_argument('-fs',
                        '--fixedspeed',
                        action='store_true',
                        help="(HUMAN ONLY) Fixes the game speed so that it does not increase with lines cleared. "
                             "Emulates the behaviour used for AI.")

    # AI (-ai or --ai) - The game will be played by an artificial intelligence
    # Possible values:
    # - 'play' - The AI will use pre-defined weights to play the game
    # - 'learn' - The AI will use DQL to learn the weights

    parser.add_argument('-ai',
                        '--ai',
                        choices=['play', 'learn'],
                        help="Sets an AI as the player ('play' for the AI to use predefined weights, "
                             "'learn' for the AI to learn using DQL)")

    # SEED (-s or --seed) - Sets the seed for all random events
    # Value is introduced by the user, and is mostly used to guarantee reproducibility during training
    # If not set, a random seed will be used instead

    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help="Sets a seed for all random events. Note that reproducibility is not totally guaranteed "
                             "due to Keras.")

    # AGENT TYPE (-at or --agenttype) - Specifies the type of agent that will be used to play the game
    # Possible values:
    # - 'standard_new'      (DEFAULT) The AI logic is the new approach used for DQL (considering that each action =>
    #                       the final position of the piece)
    # - 'prioritized_new'   A variation of the standard_new agent, where Prioritized Experience Replay is used
    # - 'random_new'        A totally random agent using the new approach. Only usable in PLAY mode
    # - 'standard_old'      The AI logic will be the standard DQL agent logic (with a basic neural network) for a
    #                       classical Q-Learning approach, considering that each action is a movement of the piece
    # - 'weighted_old'      A variation of the standard_old agent, that uses different weights when performing a
    #                       random action
    # - 'random_old'        A totally random agent using the old approach. Only usable in PLAY mode

    parser.add_argument('-at',
                        '--agenttype',
                        choices=['standard_new', 'prioritized_new', 'random_new',
                                 'standard_old', 'weighted_old', 'random_old'],
                        help="(AI ONLY) Sets the type of agent to be used. Note that agents with '-old' in the "
                             "name use the old, single-action based approach (action = actual game input), as opposed "
                             "to the approach used by the other agents (action = final position of the current piece). "
                             "DEFAULT: " + agent_type)

    # WEIGHTS (-w or --weights) - Only for AI players. Loads the neural network weights from a file, to use a trained
    # neural network. If not set, weights will not be set (and the random initialized weights will be used instead).
    # Value is introduced by the user.

    parser.add_argument('-w',
                        '--weights',
                        help="(AI PLAYING ONLY) Loads the pre-trained weights in the specified file. If not set, "
                             "the random initial weights will be used instead")

    # FAST TRAINING (-f or --fast) - Only for training. If the argument is present, the training will be done in
    # fast mode (no graphics will be displayed and time will run faster, to allow to train while in the background)

    parser.add_argument('-f',
                        '--fast',
                        action='store_true',
                        help="(LEARNING ONLY) Trains the network in fast mode (without rendering "
                             "any graphics and with sped-up time)")

    # EXPERIENCE REPLAY SIZE (-er or --experiencereplay) - Only for training. Specifies the maximum amount of
    # experiences stored in the experience replay at once.
    # Value is introduced by the user).

    parser.add_argument('-er',
                        '--experiencereplay',
                        type=int,
                        help="(LEARNING ONLY) Sets how many experiences will be taken from the experience replay at "
                             "once while learning. DEFAULT = " + str(experience_replay_size))

    # BATCH SIZE (-b or --batch) - Only for training. Specifies how many experiences are taken from the
    # experience replay to train at once.
    # Value is introduced by the user (cannot be bigger than the experience replay size).

    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        help="(LEARNING ONLY) Sets how many experiences will be taken from the experience replay at "
                             "once while learning. Cannot be bigger than the experience replay size. "
                             "DEFAULT = " + str(batch_size))

    # REWARDS METHOD (-rw or --reward) - Specified the method used to compute the reward
    # for the state/action pair.
    # Possible values:
    # - 'game': Method based directly on game score (the reward is based on the score increase of the action)
    # - 'heuristic': Method based on the quality of the board after applying an action
    # All heuristics are properly defined within the compute_reward_old function

    parser.add_argument('-rw',
                        '--reward',
                        choices=['game', 'heuristic'],
                        help="(AI ONLY) Sets the method to be used to compute the reward for a state/action pair. "
                             "DEFAULT: " + rewards_method)

    # GAMMA (-g or --gamma) - Initial value for the gamma variable (discount factor,
    # importance given to future rewards in Q-learning)
    # Value is introduced by the user.

    parser.add_argument('-g',
                        '--gamma',
                        type=float,
                        help="(LEARNING ONLY) Sets a value for the gamma variable (discount factor, importance given "
                             "to future rewards in Q-learning). DEFAULT = " + str(gamma))

    # EPSILON (-eps or --epsilon) - Initial variable for the epsilon variable (initial chance for a random action during
    # learning with Deep Q-Learning (part of the exploration-exploitation principle)
    # Value is introduced by the user (must be between 0 and 1).

    parser.add_argument('-eps',
                        '--epsilon',
                        type=float,
                        help="(LEARNING ONLY) Sets a value for the epsilon variable (initial chance to take a random "
                             "action during learning, part of exploration-exploitation). Must be between 0 and 1. "
                             "DEFAULT = " + str(epsilon))

    # EPSILON PERCENTAGE (-epp or --epsilonpercentage) - Percentage of epochs after which the epsilon value will be
    # minimum. Epsilon will decrease linearly from its initial value to the minimum value, reaching the minimum once
    # this percentage of epochs have passed. Value is introduced by the user (must be between 0 and 100)

    parser.add_argument('-epp',
                        '--epsilonpercentage',
                        type=int,
                        help="(LEARNING ONLY) Sets a percentage of epochs after which the epsilon value will be "
                             "minimum. Epsilon will be decreased linearly from the maximum at 0%% epochs to the minimum "
                             "at this%% epochs. Value must be between 0 and 100. DEFAULT = " + str(epsilon_percentage))

    # MINIMUM EPSILON (-mep or --minimumepsilon) - Minimum value of epsilon achieved. Epsilon will not go lower than
    # this value.
    # Value is introduced by the user (must be between 0 and the initial epsilon value)

    parser.add_argument('-mep',
                        '--minimumepsilon',
                        type=float,
                        help="(LEARNING ONLY) Sets the minimum value for the epsilon variable. Epsilon value will not "
                             "go below this value. Value must be between 0 and the initial epsilon value. "
                             "DEFAULT = " + str(minimum_epsilon))

    # LEARNING RATE (-lr or --learningrate) - Initial for the learning rate of the optimizer (how much new experiences
    # are valued in the neural network)
    # Value is introduced by the user.

    parser.add_argument('-lr',
                        '--learningrate',
                        type=float,
                        help="(LEARNING ONLY) Sets a value for the learning rate (value given to new samples in the "
                             "neural network). DEFAULT = " + str(learning_rate))

    # TOTAL EPOCHS (-epo or --epochs) - Total number of epochs to train the agent (the agent will be trained
    # this amount of epochs). Value is introduced by the user (must be positive)

    parser.add_argument('-epo',
                        '--epochs',
                        type=int,
                        help="(LEARNING ONLY) Sets the total amount of epochs to train the agent. Must be positive. "
                             "DEFAULT = " + str(total_epochs))

    # Parse the arguments
    arguments = vars(parser.parse_args())

    if arguments['silent']:
        sound_active = False

    if arguments['fixedspeed']:
        fixed_speed = True
        initial_speed = game_speed_ai
        speed_modifier = 0

    if arguments['ai'] is not None:
        ai_player = True
        if arguments['ai'] == 'learn':
            ai_learning = True

            # Train mode is always silent, turn off the sound
            sound_active = False

    if arguments['seed'] is not None:
        seed = arguments['seed']
        # Sets the seed
        random.seed(seed)

    if arguments['agenttype'] is not None:
        agent_type = arguments['agenttype']

    if arguments['weights'] is not None:
        weights = arguments['weights']

    if arguments['fast']:
        fast_training = True

    if arguments['experiencereplay'] is not None:
        experience_replay_size = arguments['experiencereplay']

    if arguments['batchsize'] is not None:
        batch_size = arguments['batchsize']
        if batch_size < 1 or batch_size > experience_replay_size:
            print("INVALID VALUE: Batch size must be between 1 and the maximum experience replay size. Passed: " + str(batch_size))
            sys.exit()

    if arguments['reward'] is not None:
        rewards_method = arguments['reward']

    if arguments['gamma'] is not None:
        gamma = arguments['gamma']

    if arguments['epsilon'] is not None:
        epsilon = arguments['epsilon']
        if epsilon < 0.0 or epsilon > 1.0:
            print("INVALID VALUE: Epsilon  must be between 0.0 and 1.0. Passed: " + str(epsilon))
            sys.exit()

    if arguments['epsilonpercentage'] is not None:
        epsilon_percentage = arguments['epsilonpercentage']
        if epsilon_percentage < 0 or epsilon_percentage > 100:
            print("INVALID VALUE: Epsilon percentage must be between 0% and 100%. Passed: " + str(epsilon_percentage))
            sys.exit()

    if arguments['minimumepsilon'] is not None:
        minimum_epsilon = arguments['minimumepsilon']
        if minimum_epsilon < 0.0 or minimum_epsilon > epsilon:
            print("INVALID VALUE: Minimum epsilon must be between 0.0 and the specified initial value of epsilon (" + str(epsilon) + "). Passed: " + str(epsilon_percentage))
            sys.exit()

    if arguments['learningrate'] is not None:
        learning_rate = arguments['learningrate']

    if arguments['epochs'] is not None:
        total_epochs = arguments['epochs']
        if total_epochs <= 0:
            print("INVALID VALUE: Total number of epochs must be positive. Passed: " + str(total_epochs))
            sys.exit()

    # Initialize pygame
    pygame.font.init()
    pygame.mixer.pre_init(22050, -16, 2, 32)
    pygame.init()
    pygame.mixer.init()

    # Prepare the game window
    # (the window will be wider if an AI player is active, but only if fast mode is not active)
    if ai_learning and fast_training:
        win = None
    elif ai_player:
        win = pygame.display.set_mode((screen_width + screen_width_extra, screen_height))
    else:
        win = pygame.display.set_mode((screen_width, screen_height))

    pygame.display.set_caption("DQL - TETRIS")

    # If the sound is active, load the sounds (no need to otherwise)
    if sound_active:
        sound_gallery = prepare_sounds([
            ("action", "beep.wav"),
            ("fall", "fall.wav"),
            ("line", "lineclear.ogg"),
            ("lost", "lost.ogg")])

    # Start the appropriate logic, depending on the type of player (human, AI learning or AI playing)

    # Check if there is an AI player
    if ai_player:

        # Compute what the epsilon decay would actually be
        # The key idea is that epsilon reaches 0 once the specified % of epochs is reached
        epochs_to_reduce = (total_epochs * epsilon_percentage) / 100
        epsilon_decay = (epsilon - minimum_epsilon) / epochs_to_reduce

        # Ensure first a proper value of epsilon: all epsilon values should be 0 ONLY if the AI is a player
        # (we want to strictly follow the policy)
        if not ai_learning:
            epsilon = 0
            epsilon_decay = 0

        # Instantiate the appropriate agent
        if agent_type == 'standard_new':
            agent = dql_agent_new.DQLAgentNew(learning_rate,
                                              gamma,
                                              epsilon,
                                              epsilon_decay,
                                              minimum_epsilon,
                                              batch_size,
                                              total_epochs,
                                              experience_replay_size,
                                              seed,
                                              rewards_method)
        elif agent_type == 'prioritized_new':
            agent = prioritized_agent_new.PrioritizedAgentNew(learning_rate,
                                                              gamma,
                                                              epsilon,
                                                              epsilon_decay,
                                                              minimum_epsilon,
                                                              batch_size,
                                                              total_epochs,
                                                              experience_replay_size,
                                                              seed,
                                                              rewards_method)
        elif agent_type == 'random_new':
            agent = random_agent_new.RandomAgentNew(seed)
        elif agent_type == 'standard_old':
            agent = dql_agent_old.DQLAgentOld(learning_rate,
                                              gamma,
                                              epsilon,
                                              epsilon_decay,
                                              minimum_epsilon,
                                              batch_size,
                                              total_epochs,
                                              experience_replay_size,
                                              seed,
                                              rewards_method)
        elif agent_type == 'weighted_old':
            agent = weighted_agent_old.WeightedAgentOld(learning_rate,
                                                        gamma,
                                                        epsilon,
                                                        epsilon_decay,
                                                        minimum_epsilon,
                                                        batch_size,
                                                        total_epochs,
                                                        experience_replay_size,
                                                        seed,
                                                        rewards_method)
        elif agent_type == 'random_old':
            agent = random_agent_old.RandomAgentOld(seed)

    # If the game is in learning mode, directly launch the game (without the main menu)
    if ai_learning:
        # Before starting training, prepare the training structure
        agent.initialize_learning_structure()
        # Start the appropiate main train loop
        if agent.agent_type == "old":
            main_ai_learn_old(win)
        else:
            main_ai_learn_new(win)
    # Otherwise, launch the main menu
    else:
        # If there is an agent, try to load the weights
        if ai_player:
            agent.load_weights(weights)
        menu_logic(win)
