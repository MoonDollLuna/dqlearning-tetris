import pygame
import random
import sys

# Based on the following tutorial:
# https://techwithtim.net/tutorials/game-development-with-python/tetris-pygame/tutorial-1/

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

pygame.font.init()

# TODO: variables globales en mayus?
# Global variables

# Window size
s_width = 800
s_height = 700

# Playzone size
play_width = 300  # 300 // 10 = 30 width per block
play_height = 600  # 600 // 20 = 30 height per block

# Playzone position
top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

# Block size
block_size = 30


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
      '0000.',
      '.....',
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

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0


# TODO: ESTO ES SUPER CUTRE!!!!!
def max_score():
    with open('scores.txt', 'r') as f:
        lines = f.readlines()
        score = lines[0].strip

    return score


# TODO: CUUUUUTREEEEEEEEEEEEE
def update_score(nscore):

    score = max_score()

    with open('scores.txt', 'w') as f:
        if int(score) > nscore:
            f.write(str(score))
        else:
            f.write(str(nscore))


def create_grid(locked_positions={}):
    # Grid (playzone) is represented as a matrix of colours
    # Initially all colors (empty positions) are black
    grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]

    # For all fixed blocks (locked positions), color the corresponding position
    # to the appropiate color
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
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
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


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("comicsans", size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width()/2)), top_left_y + play_height/2 - (label.get_height()/2))


# Draws the grid on the playzone
# TODO: OPTIMIZA EL CODIGO (dibujalas por separado)
def draw_grid(surface, grid):
    for i in range(len(grid)):
        pygame.draw.line(surface, (128, 128, 128), (top_left_x, top_left_y + i*block_size),
                         (top_left_x + play_width, top_left_y + i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (top_left_x + j*block_size, top_left_y),
                             (top_left_x + j*block_size, top_left_y + play_height))


# TODO: reinicia reloj cuando limpias un row?
# TODO: añadir efecto visual
def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
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


# TODO: saca el font afuera y usalo personalizado
def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('NEXT SHAPE', 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100

    shape_format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*block_size, sy + i * block_size, block_size, block_size), 0)

    surface.blit(label, (sx + 10, sy - 30))


def draw_window(surface, grid, score=0):

    # Fills the background (black color)
    surface.fill((0,0,0))

    # TODO: Cambia la font a una font arcade?
    # (con pygame.font.Font se le puede pasar como parametro)
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render("TETRIS", 1, (255,255,255))

    surface.blit(label, (top_left_x + play_width / 2 - label.get_width() / 2, 30))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 4)

    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('SCORE: ' + str(score), 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100

    surface.blit(label, (sx + 10, sy + 200))

    draw_grid(surface, grid)

    # TODO ELIMINA ESTO
    # pygame.display.update()

def main(win):
    
    locked_positions = {}
    # TODO: ESTO ASI NO ME LUCE
    high_score = max_score()

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
                    pass

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


        # TODO: Funcion general manager que dibuja TODO en la ventana
        # y luego lo actualiza de una (para evitar este plataco de espagueti)
        draw_window(win, grid, score)
        draw_next_shape(next_piece, win)
        pygame.display.flip()

        if check_lost(locked_positions):
            # TODO: AJUSTA ESTO QUE NO ME CONVENCE
            draw_text_middle(win, "GAME OVER", 80, (255, 255, 255))
            pygame.display.quit()
            pygame.time.delay(1500)
            run = False
            update_score(score)

def main_menu(win):
    run = True
    while run:
        win.fill((0,0,0))
        draw_text_middle(win, "PRESS ANY KEY TO PLAY", 60, (255, 255, 255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main(win)

    pygame.display.quit()
    pygame.quit()
    sys.exit()


win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption("DQL - TETRIS")
main_menu(win)  # start game