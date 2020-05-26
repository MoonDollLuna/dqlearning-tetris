# CODE BY: Luna Jiménez Fernández
# Originally based on the following tutorial:
# https://techwithtim.net/tutorials/game-development-with-python/tetris-pygame/tutorial-1/

# The game consists of a 10 x 20 grid (standard tetris size)

###########
# IMPORTS #
###########

import sys
import argparse
import copy
import random
import os.path

import numpy as np

# Import used for our own agent scripts
from agents import dql_agent

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

# Maximum number of lines (if an agent reaches this amount of lines in a game, the game is cut)
max_lines_training = 100

####################
# PLAYER VARIABLES #
####################

# All of these variables can be set using arguments while launching the script
# However, they're stored here to give them default values

# Whether the sound is active (TRUE) or not (FALSE)
# Set to false using --silent
sound_active = True

# If the player is human (FALSE) or an AI (TRUE)
# Set using --ai play or --ai learn
ai_player = False

# If the AI player is in playing (FALSE) or learning (TRUE) mode
# Set using --ai learn
ai_learning = False

# Set seed for reproducibility. NONE if no seed has been set
# Set using --seed
seed = None

# Set type of agent to be used. Default value is Standard. Only relevant when using AI (training or playing)
# Set using --agenttype
agent_type = 'standard'

# TRAINING VARIABLES - These variables are only relevant while ai_learning is TRUE #

# If the training is being done in normal, visual mode (FALSE) or in fast, text-only mode (TRUE)
# Set using --fast
fast_training = False

# Batch size used to sample from the Experience Replay. Default value is specified below
# Set using --batchsize
batch_size = 512

# Gamma value used by DQL (learning rate of DQL). Default value is specified below
# Set using --gamma
gamma = 0.6

# Epsilon value used by DQL (chance to perform a random action in exploration-exploitation)
# Default value is specified below
# Set using --epsilon
epsilon = 0.85

# Epsilon decay used by DQL (how much does epsilon decay every epoch, multiplicatively)
# Default value is specified below
# Set using --epsilondecay
epsilon_decay = 0.995

# Minimum epsilon value used by DQL (epsilon cannot go below this value)
# Default value is specified below
# Set using --minimumepsilon
min_epsilon = 0.1

# Learning rate used by the neural network. Default value is specified below
# Set using --learningrate
learning_rate = 0.01

# Number of epochs used to train the agent. Default value is specified below.
# Set using --epochs
total_epochs = 1000


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


#################
# AGENT METHODS #
#################

# GRAPHICS #

def draw_ai_player_information(surface, current_state, q_values):
    """
    Draws relevant information for the AI player (in order to visualize the choices being taken)

    :param surface: Surface in which to draw the information.
    :param current_state: Last state received by the agent, to be displayed on screen
    :param q_values: Q-Value of every action
    """

    # Create a rectangle for the additional HUD
    pygame.draw.rect(surface, background_color, (screen_width, 0, screen_width_extra, screen_height), 0)
    pygame.draw.rect(surface, playground_border_color, (screen_width, 0, screen_width_extra, screen_height), 5)

    # Identify where to place the additional AI HUD
    hud_begin_x = screen_width + 25

    # Create two fonts
    small_font = pygame.font.Font(font_path, 20)
    big_font = pygame.font.Font(font_path, 30)

    # STATE
    state_y = 15

    # Write the state title and print it
    state_text = big_font.render('STATE', 1, (0, 0, 0))
    surface.blit(state_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - state_text.get_width() / 2,
                              state_y + 25 - state_text.get_height() / 2))

    # Draw the state itself

    # Initial positions and size of each block
    state_initial_x = screen_width + 75
    state_initial_y = 70
    state_block_size = 15

    # Draw a rectangle behind the state
    pygame.draw.rect(surface, playground_border_color, (state_initial_x - 5, state_initial_y - 5, state_block_size * 10 + 10, state_block_size * 20 + 10), 0)

    # Loop through all elements of the state
    for (y, x), element in np.ndenumerate(current_state):

        # Set the color depending on the value of the element
        # (0 is black, 1 is gray, 2 is white)
        if element == 0:
            color = (0, 0, 0)
        elif element == 1:
            color = (128, 128, 128)
        else:
            color = (255, 255, 255)

        # Draw the actual square in the appropiate position
        pygame.draw.rect(surface,
                         color,
                         (state_initial_x + x * state_block_size,
                          state_initial_y + y * state_block_size,
                          state_block_size,
                          state_block_size),
                         0)

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
    right_text = small_font.render('RIGHT:', 1, (0, 0, 0))
    surface.blit(right_text, (hud_begin_x, qvalues_y + 50))

    # Right content
    right_content_text = small_font.render(str(q_values[0][0]), 1, (0, 0, 0))
    surface.blit(right_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - right_content_text.get_width() / 2, qvalues_y + 80))

    # LEFT
    left_text = small_font.render('LEFT:', 1, (0, 0, 0))
    surface.blit(left_text, (hud_begin_x, qvalues_y + 110))

    # Left content
    left_content_text = small_font.render(str(q_values[0][1]), 1, (0, 0, 0))
    surface.blit(left_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - left_content_text.get_width() / 2, qvalues_y + 140))

    # ROTATE
    rotate_text = small_font.render('ROTATE:', 1, (0, 0, 0))
    surface.blit(rotate_text, (hud_begin_x, qvalues_y + 170))

    # Rotate content
    rotate_content_text = small_font.render(str(q_values[0][2]), 1, (0, 0, 0))
    surface.blit(rotate_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - rotate_content_text.get_width() / 2, qvalues_y + 200))

    # HARD DROP
    harddrop_text = small_font.render('HARD DROP:', 1, (0, 0, 0))
    surface.blit(harddrop_text, (hud_begin_x, qvalues_y + 230))

    # Rotate content
    harddrop_content_text = small_font.render(str(q_values[0][3]), 1, (0, 0, 0))
    surface.blit(harddrop_content_text, (hud_begin_x + ((screen_width + screen_width_extra) - hud_begin_x - 25) // 2 - harddrop_content_text.get_width() / 2, qvalues_y + 260))


# GAMEPLAY #

def generate_state(locked_positions, current_piece):
    """
    Computes the current state from the current game grid.

    The state will be store as a 20x10 numpy matrix, where each cell can have one of the following values:
    - 0: empty
    - 1: occupied
    - 2: occupied by current piece

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

    # Obtain the positions of the current piece and change them to 2
    piece_positions = generate_shape_positions(current_piece)
    for (x, y) in piece_positions:
        # Ignore negative positions (they're still out of bounds)
        if x >= 0 and y >= 0:
            grid[y][x] = 2

    return np.array(grid)


def compute_reward(game_finished, piece_locked, lines_cleared, lowest_position_filled):
    """
    Computes the reward for a pair of state, action taking into account some details

    The reward computed is as follows:
    * If the game is finished, REWARD = -10 (we don't want the agent to lose)
    * If no piece has been locked, REWARD = -0.1 (we want an incentive for the agent to lock pieces)
    * If a piece has been locked:
        * No line cleared, REWARD = lowest_position_filled / 10 (locking pieces is good, the deeper the better)
        * Lines cleared, REWARD = +2^(lines_cleared + 1) (the more lines that are cleared at once, the better the state is)

    :param game_finished: TRUE if the action caused the end of the game, FALSE otherwise
    :param piece_locked: TRUE if the piece has been locked, FALSE otherwise
    :param lines_cleared: (Only if piece_locked = TRUE) How many lines were cleared with the locked piece.
    :param lowest_position_filled: (Only if piece_locked = TRUE) The lowest Y value reached by the locked piece.
    :return: The reward for the pair state, action
    """

    # TODO: AJUSTA REWARD (HACIENDO UNA OPCION ALTERNATIVA) QUE VALORE MEJOR O PEOR EL COLOCAR LA FICHA DEPENDIENDO DE ALGUNOS ATRIBUTOS

    # Game finished: immediate big penalty
    if game_finished:
        return -10
    # Game not finished but piece not locked: very small penalty (want the agent to try to lock fast)
    elif not piece_locked:
        return -0.1
    else:
        # 0 lines locked: reward the agent according to how deep the piece is (lower = better)
        if lines_cleared == 0:
            return lowest_position_filled / 10
        # 1 or more lines locked: give the agent a bigger reward (more lines = better reward)
        else:
            return 2 ** (lines_cleared + 1)


###############################
# MAIN LOOP AUXILIARY METHODS #
###############################

# These methods are the main loop methods shared between all instances of the main game loop
# (human player, AI player or AI learner)
# TODO REVISA ESTOS METODOS JULIA A VER QUE TE PARECE

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

            # Update lines, level and speed
            lines_cleared = clear_rows(grid, locked_positions)
            lines += len(lines_cleared)
            level = lines // 10

            current_speed = initial_speed - speed_modifier * level
            # Ensure that the speed doesn't go below a limit
            if current_speed < minimum_speed:
                current_speed = minimum_speed

            # Play the piece lock sound
            play_sound("fall")

            # Update score
            if len(lines_cleared) == 0:
                # No lines cleared
                score += shape_y + 1
            else:
                # One or more lines cleared

                # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                play_sound("line")

                # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                draw_clear_row(win, lines_cleared)

                # Compute the score to add
                multiplier = 0
                buffer = len(lines_cleared)
                # This increases exponentially the multiplier depending on lines
                # 1 line = x1, 2 lines = x3, 3 lines = x6, 4 lines = x10
                while buffer > 0:
                    multiplier += buffer
                    buffer -= 1
                score += (level + 1) * multiplier * 100

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


def main_ai_player(win):
    """
    Main logic of the game when a AI player is active

    The main differences are that the actions are now polled from the agent instead of from the user inputs

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
    q_values = np.array([[0, 0, 0, 0]])

    # While the game is not over (main logic loop)
    while run:

        # Create the grid and update all the clocks, marking a new tick
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        poll_time += clock.get_rawtime()
        clock.tick()

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
            process_inputs([action], current_piece, grid)

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

            # Update lines and level (speed is not updated)
            lines_cleared = clear_rows(grid, locked_positions)
            lines += len(lines_cleared)
            level = lines // 10

            # Play the piece lock sound
            play_sound("fall")

            # Update score
            if len(lines_cleared) == 0:
                # No lines cleared
                score += shape_y + 1
            else:
                # One or more lines cleared

                # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                play_sound("line")

                # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                draw_ai_player_information(win, current_state, q_values)
                draw_clear_row(win, lines_cleared)

                # Compute the score to add
                multiplier = 0
                buffer = len(lines_cleared)
                # This increases exponentially the multiplier depending on lines
                # 1 line = x1, 2 lines = x3, 3 lines = x6, 4 lines = x10
                while buffer > 0:
                    multiplier += buffer
                    buffer -= 1
                score += (level + 1) * multiplier * 100

            # Prepare everything for the next loop
            change_piece = False
            clock.tick()

        # Draw everything (original HUD and AI HUD)
        draw_manager(win, grid, current_piece, next_piece, score, level, lines)
        draw_ai_player_information(win, current_state, q_values)
        # Update the screen
        pygame.display.flip()

        # Check if the game has ended
        if check_defeat(locked_positions):
            stop_sounds()
            play_sound("lost")
            draw_game_over_effect(win)
            run = False


def main_ai_learn(win):
    """
    Main logic of the game when the AI is in learning mode

    This method has several differences compared to the standard AI player loop:
    - The game automatically replays itself after every epoch
    - The game can run either on a real clock or a logical clock (when running in fast mode)

    :param win: Surface used to draw all the elements.
    """

    # Initialize the epoch counter
    current_epoch = 1

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

        # AI: Keep the Q-Values outside of the loop (to keep their values through iterations of the loop)
        # They are initialized with placeholder values
        q_values = np.array([[0, 0, 0, 0]])

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
                process_inputs([action], current_piece, grid)

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

                # Update lines and level (speed will not be updated)
                lines_cleared = clear_rows(grid, locked_positions)
                lines += len(lines_cleared)
                level = lines // 10

                # Update the lines cleared outside
                lines_cleared_store = len(lines_cleared)

                # Play the piece lock sound
                play_sound("fall")

                # Update score
                if len(lines_cleared) == 0:
                    # No lines cleared
                    score += shape_y + 1
                else:
                    # One or more lines cleared

                    # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                    play_sound("line")

                    # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                    if not fast_training:
                        draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                        draw_clear_row(win, lines_cleared)

                    # Compute the score to add
                    multiplier = 0
                    buffer = len(lines_cleared)
                    # This increases exponentially the multiplier depending on lines
                    # 1 line = x1, 2 lines = x3, 3 lines = x6, 4 lines = x10
                    while buffer > 0:
                        multiplier += buffer
                        buffer -= 1
                    score += (level + 1) * multiplier * 100

                # Prepare everything for the next loop
                change_piece = False

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
                reward = compute_reward(not run, final_state, lines_cleared_store, lowest_y)

                # Store the experience into the agent
                agent.insert_experience(current_state, action, reward, new_state, final_state)

            # Draw everything (original HUD and AI HUD) IF not in fast mode
            if not fast_training:
                draw_manager(win, grid, current_piece, next_piece, score, level, lines)
                #draw_ai_player_information(win, current_state, q_values)
                # Update the screen
                pygame.display.flip()

        # GAME END

        # Advance to the next epoch
        current_epoch += 1

        # Notify the agent of the new epoch
        agent.finish_epoch(lines, score)

def OLD(win):

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
    # (AI player uses a real-time clock)
    clock = pygame.time.Clock()
    fall_time = 0
    # Poll time is used to check for the AI player actions
    poll_time = 0

    # Start playing the song (if sound is active)
    play_song()

    # While the game is not over (main logic loop)
    while run:

        # Create the grid and update all the clocks, marking a new tick
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

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

                # Left key (move left and play the appropriate sound)
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1
                    play_sound("action")

                # Right key (move right and play the appropriate sound)
                if event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                    play_sound("action")

                # Down key (soft drop)
                if event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

                # Up key (hard/instant drop)
                if event.key == pygame.K_UP:

                    # Try to move the piece down until an illegal position is reached, and then move upwards to reach
                    # the final valid position
                    while valid_space(current_piece, grid):
                        current_piece.y += 1
                    current_piece.y -= 1

                    # After a hard drop, piece will be guaranteed to be locked
                    change_piece = True

                # R key (rotation and play the appropriate sound)
                if event.key == pygame.K_r:
                    current_piece.rotation += 1
                    if not valid_space(current_piece, grid):
                        current_piece.rotation -= 1
                    play_sound("action")

        # Clock calculations (piece falling)
        if fall_time > current_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        # Place the current piece into the grid
        shape_pos = generate_shape_positions(current_piece)
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

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

            # Update lines, level and speed
            lines_cleared = clear_rows(grid, locked_positions)
            lines += len(lines_cleared)
            level = lines // 10

            current_speed = initial_speed - speed_modifier * level
            # Ensure that the speed doesn't go below a limit
            if current_speed < minimum_speed:
                current_speed = 0.05

            # Play the piece lock sound
            play_sound("fall")

            # Update score
            if len(lines_cleared) == 0:
                # No lines cleared
                score += shape_y + 1
            else:
                # One or more lines cleared

                # Play the appropriate sound. Sound is played before the effect is drawn to ensure it's not delayed
                play_sound("line")

                # Draw the screen first (to ensure the piece is displayed on its proper place) and then draw the effect
                draw_manager(win, grid, current_piece, next_piece, score, level, lines - len(lines_cleared))
                draw_clear_row(win, lines_cleared)

                # Compute the score to add
                multiplier = 0
                buffer = len(lines_cleared)
                # This increases exponentially the multiplier depending on lines
                # 1 line = x1, 2 lines = x3, 3 lines = x6, 4 lines = x10
                while buffer > 0:
                    multiplier += buffer
                    buffer -= 1
                score += (level + 1) * multiplier * 100

            # Prepare everything for the next loop
            change_piece = False
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
                # Else, start the appropriate game logic depending on the type of player
                else:
                    if not ai_player:
                        main_human_player(win)
                        pygame.event.clear()
                    else:
                        main_ai_player(win)
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
    # - 'standard' - (DEFAULT) The AI logic will be the standard DQL agent logic (with a basic neural network)

    parser.add_argument('-at',
                        '--agenttype',
                        choices=['standard'],
                        help="(AI ONLY) Sets the type of agent to be used. DEFAULT: Standard")

    # FAST TRAINING (-f or --fast) - Only for training. If the argument is present, the training will be done in
    # fast mode (no graphics will be displayed and time will run faster, to allow to train while in the background)

    parser.add_argument('-f',
                        '--fast',
                        action='store_true',
                        help="(LEARNING ONLY) Trains the network in fast mode (without rendering "
                             "any graphics and with sped-up time)")

    # BATCH SIZE (-b or --batch) - Only for training. Specifies how many experiences are taken from the
    # experience replay to train at once.
    # Value is introduced by the user (must be between 1 and 2000).

    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        help="(LEARNING ONLY) Sets how many experiences will be taken from the experience replay at "
                             "once while learning. Must be between 1 and 2000. DEFAULT = " + str(batch_size))

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

    # EPSILON DECAY (-ed or --epsilondecay) - Decay used for epsilon. Epsilon is reduced multiplicatively after
    # every epoch, by multiplying it by this value. Value is introduced by the user (must be between 0 and 1).

    parser.add_argument('-ed',
                        '--epsilondecay',
                        type=float,
                        help="(LEARNING ONLY) Sets the value for the epsilon decay (how much does the epsilon value"
                             " decrease after every epoch, obtained by multiplying epsilon by this value). "
                             "Must be between 0 and 1. "
                             "DEFAULT = " + str(epsilon_decay))

    # MINIMUM EPSILON (-me or --minimumepsilon) - Minimum value for epsilon (epsilon cannot be reduced below this value
    # by the decay). Value is introduced by the user (must be between 0 and 1)

    parser.add_argument('-me',
                        '--minimumepsilon',
                        type=float,
                        help="(LEARNING ONLY) Sets the minimum value for epsilon (epsilon cannot decay below this "
                             "value). Must be between 0 and 1. "
                             "DEFAULT = " + str(min_epsilon))

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

    # There is no need to store fixed speed: the speed can be changed directly
    if arguments['fixedspeed']:
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

    if arguments['fast']:
        fast_training = True

    if arguments['batchsize'] is not None:
        batch_size = arguments['batchsize']
        if batch_size < 1 or batch_size > 2000:
            raise ValueError("Batch size must be between 1 and 2000")

    if arguments['gamma'] is not None:
        gamma = arguments['gamma']

    if arguments['epsilon'] is not None:
        epsilon = arguments['epsilon']
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be between 0.0 and 1.0")

    if arguments['epsilondecay'] is not None:
        epsilon_decay = arguments['epsilondecay']
        if epsilon_decay < 0.0 or epsilon_decay > 1.0:
            raise ValueError("Epsilon decay must be between 0.0 and 1.0")

    if arguments['minimumepsilon'] is not None:
        min_epsilon = arguments['minimumepsilon']
        if min_epsilon < 0.0 or min_epsilon > 1.0:
            raise ValueError("Minimum epsilon must be between 0.0 and 1.0")

    if arguments['learningrate'] is not None:
        learning_rate = arguments['learningrate']

    if arguments['epochs'] is not None:
        total_epochs = arguments['epochs']
        if total_epochs <= 0:
            raise ValueError("Total number of epochs must be positive")

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

        # Ensure first a proper value of epsilon: all epsilon values should be 0 ONLY if the AI is a player
        # (we want to strictly follow the policy)
        if not ai_learning:
            epsilon = 0
            epsilon_decay = 0
            min_epsilon = 0

        # AI player present: instantiate the appropriate agent
        # A switch case statement would be used here, but since Python does not implement it,
        # if elses will be used instead
        if agent_type == 'standard':
            agent = dql_agent.DQLAgent(learning_rate,
                                       gamma,
                                       epsilon,
                                       epsilon_decay,
                                       min_epsilon,
                                       batch_size,
                                       seed)

    # If the game is in learning mode, directly launch the game (without the main menu)
    if ai_learning:
        # Start the main train loop
        main_ai_learn(win)
    # Otherwise, launch the main menu
    else:
        menu_logic(win)
