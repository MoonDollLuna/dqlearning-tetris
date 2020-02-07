# CODE BY: Luna Jiménez Fernández
# Based on the following tutorial:
# https://techwithtim.net/tutorials/game-development-with-python/tetris-pygame/tutorial-1/

# Imports
import pygame
import random
import sys
import argparse
import copy

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

####################
# Global variables #
####################

# Window size
screen_width = 700
screen_height = 800

# Block size
block_size = 40

# Playzone size
play_width = 10 * block_size # 10 block-wide playzone
play_height = 20 * block_size # 20 block-high playzone

# Playzone position
top_left_x = 20 + block_size // 2
top_left_y = 0

# Path to the font
font_path = "./fonts/ARCADE_N.TTF"

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

# Used colors
shape_colors = [(0, 240, 0), (240, 0, 0), (0, 240, 240), (240, 240, 0), (240, 160, 0), (0, 0, 240), (160, 0, 240)]
background_color = (170, 170, 170)
piece_border_color = (0, 0, 0)
playground_border_color = (75, 75, 75)

#####################
# CLASS DEFINITIONS #
#####################


class Piece(object):
    def __init__(self, x, y, shape):
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

    # TODO: ESTO NO VA Y NO TENGO CLARO POR QUE
    # Clone the shape
    shadow_shape = copy.deepcopy(shape)

    # Find the position where the piece would be
    while valid_space(shadow_shape, grid):
        shadow_shape.y += 1
    shadow_shape.y -= 1

    # Draw all the blocks in the appropiate color
    for position in convert_shape_format(shadow_shape):
        pygame.draw.rect(surface, shadow_shape.color, (top_left_x + position[0] * block_size, top_left_y + position[1] * block_size, block_size, block_size), 5)


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
    pygame.draw.rect (surface, background_color, (hud_begin_x, hud_begin_y + 20, screen_width - hud_begin_x - 10, 60), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x, hud_begin_y + 20, screen_width - hud_begin_x - 10, 60), 5)

    # Draw the title and place it
    font = pygame.font.Font(font_path, 20)
    text = font.render('NEXT SHAPE', 1, (0, 0, 0))
    surface.blit(text, (hud_begin_x + (screen_width - hud_begin_x - 10) // 2 - text.get_width() / 2, hud_begin_y + 50 - text.get_height() / 2))

    # Draw a rectangle to contain the shape below
    pygame.draw.rect(surface, background_color, (hud_begin_x + 10, hud_begin_y + 90, screen_width - hud_begin_x - 30, 180), 0)
    pygame.draw.rect(surface, playground_border_color, (hud_begin_x + 10, hud_begin_y + 90, screen_width - hud_begin_x - 30, 180), 5)

    # Draws the shape
    sx = top_left_x + play_width + 40
    sy = hud_begin_y + 100

    shape_format = shape.shape[shape.rotation % len(shape.shape)]

    # TODO: POSIBLEMENTE SE PUEDE AJUSTAR ESTO
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


####################
# GAMEPLAY METHODS #
####################

def create_grid(locked_positions={}):
    # Grid (playzone) is represented as a matrix of colours
    # Initially all colors (empty positions) are black
    grid = [[background_color for _ in range(10)] for _ in range(20)]

    # For all fixed blocks (locked positions), color the corresponding position
    # to the appropriate color
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c

    return grid


def convert_shape_format(shape):
    positions = []

    # Identify current shape
    shape_format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    # TODO: CALCULA MEJOR EL OFFSET? ESTO ES MUY CUTRE
    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == background_color] for i in range(20)]
    # Flatten the matrix
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            # Ignore new tetraminos
            # (still falling, Y > -1)
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        if pos[1] < 1:
            return True
    return False


# Returns a random shape
# TODO: Haz bag randomizer
def get_shape():
    return Piece(5, 0, random.choice(shapes))


# TODO: ESTA FUNCION SE PUEDE READAPTAR
def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("comicsans", size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width()/2), top_left_y + play_height/2 - (label.get_height()/2)))


# TODO: reinicia reloj cuando limpias un row?
# TODO: añadir efecto visual
def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if background_color not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue

    if inc > 0:
        # Explored in reverse order
        # (otherwise, we would crush previous positions)
        for key in sorted(list(locked), key= lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc


def main(win):
    
    locked_positions = {}
    # TODO: ESTO ASI NO ME LUCE
    # high_score = max_score()

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    # TODO: Ajusta el fall_speed
    fall_speed = 0.27
    level_time = 0
    # TODO: Ajusta el score
    score = 0

    while run:

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        # TODO AJUSTA ESTO
        if level_time/1000 > 5:
            level_time = 0
            if fall_speed > 0.12:
                fall_speed -= 0.005

        # TODO: CREO QUE AQUI DARA PROBLEMAS (quizas interesara moverlo abajo?)
        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():

            # TODO: Ajustar esto, que seguramente habra que cambiarlo
            # (la IA tiene que poder entrenar automaticamente)
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

            # TODO: Entiende mejor el codigo
            # TODO: Ajusta esto (permitimos pulsaciones seguidas)
            # TODO: No estartía de más poder cerrar el juego con la tecla ESC
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                if event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1

                if event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

                # Caida rapida
                if event.key == pygame.K_UP:
                    # TODO: Comentar mejor?
                    while valid_space(current_piece, grid):
                        current_piece.y += 1

                    current_piece.y -= 1
                    change_piece = True

                if event.key == pygame.K_r:
                    current_piece.rotation += 1
                    if not valid_space(current_piece, grid):
                        current_piece.rotation -= 1

        shape_pos = convert_shape_format(current_piece)

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += clear_rows(grid, locked_positions)

        # Draw everything and update the screen
        draw_manager(win, grid, current_piece, next_piece, score)
        pygame.display.flip()

        if check_lost(locked_positions):
            # TODO: AJUSTA ESTO QUE NO ME CONVENCE
            draw_text_middle(win, "GAME OVER", 80, (255, 255, 255))
            pygame.time.delay(1500)
            run = False
            # update_score(score)


def main_menu(win):
    run = True
    while run:
        win.fill((0, 0, 0))
        draw_text_middle(win, "PRESS ANY KEY TO PLAY", 60, (255, 255, 255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main(win)
                pygame.event.clear()

    pygame.display.quit()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    pygame.font.init()
    win = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("DQL - TETRIS")
    main_menu(win)  # start game