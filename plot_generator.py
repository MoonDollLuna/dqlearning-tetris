# CODE BY: Luna Jiménez Fernández

# This scripts transforms the specified .csv files into several point graphs.
# To be more precise, the graphs generated are:
#       * Score graph (score over all the epochs)
#       * Lines cleared graph (lines cleared over all the epochs)
#       * Actions graph (actions taken over all the epochs)
# Several parameters can be adjusted using arguments, specified below

# IMPORTS #

import argparse
import sys
import csv
from os.path import splitext

import matplotlib.pyplot as plt
import numpy as np

# MAIN SCRIPT #

# Create the argparser
parser = argparse.ArgumentParser(description="Creates several graphs from the specified .csv files containing info "
                                             "about the learning process of the agents.\n"
                                             "The graphs created are:\n"
                                             "\tLines cleared graph\n"
                                             "\tScore graph\n"
                                             "\tActions taken graph")

# Creates the arguments for the argparser

# FILE (-f or --file) - Loads a file into the plot. Usage: --file filename <legend name> WHERE:
#       * filename: Name of the .cvs file containing the data (must include the .csv extension)
#       * legend_name: OPTIONAL. If specified, this name will be used as the name for this info when plotting the graph.
#         Otherwise, the file name will be used.
# This argument can be called several times, to load several files at once.

parser.add_argument('-f',
                    '--file',
                    action='append',
                    nargs='+',
                    metavar=('filename', 'legend_name'),
                    help="Loads the data from a .csv file to plot a graph. Can be called multiple times. "
                         "If specified, legend_name will be used as the name in the graph. "
                         "Otherwise, the filename will be used instead (without .csv).")

# TITLE (-t or --title) - Specifies a title for the graph (mostly used to specify seed, epochs, epsilon, gamma...)
# This title will be appended to the specific title for each graph.

parser.add_argument('-t',
                    '--title',
                    help="Specifies a title to be used by all graphs. This title will be appended to the graphs own "
                         "titles.")

# Parse the arguments
arguments = vars(parser.parse_args())

files_loaded = arguments['file']
if len(files_loaded) <= 0:
    print("ERROR: At least one file needs to be loaded.")
    sys.exit()

title = arguments['title']

# Pre-process all CSV files
epochs = []
lines = []
scores = []
actions = []

for elements in files_loaded:
    # Extract the file name
    file_name = elements[0]
    legend_name = splitext(file_name)[0]

    # If a name for the legend exists, extract it
    if len(elements) > 1:
        legend_name = elements[1]

    # Open the file and read the all columns
    internal_epochs = []
    internal_lines = []
    internal_scores = []
    internal_actions = []

    with open(file_name, 'r') as file:
        # Use a csv reader
        rows = csv.reader(file, delimiter=',')
        # Ignore the titles
        next(rows, None)
        for row in rows:
            internal_epochs.append(int(row[0]))
            internal_lines.append(int(row[2]))
            internal_scores.append(int(row[1]))
            internal_actions.append(int(row[3]))

    # Insert the elements
    epochs.append((legend_name, internal_epochs))
    lines.append((legend_name, internal_lines))
    scores.append((legend_name, internal_scores))
    actions.append((legend_name, internal_actions))

# Plot all relevant graphs

# LINES:

# Create the figure
plt.figure(figsize=(12,8))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_lines) in zip(epochs, lines):
    plt.plot(internal_epochs, internal_lines, label=legend_name)

# Add the title and the axes title
plt.title('LINEAS ELIMINADAS' if title is None else ('LINEAS ELIMINADAS (' + title + ')'))
plt.xlabel('Epochs realizados')
plt.ylabel('Lineas eliminadas')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Specify the ticks for the Y axis (fixed to 100)
plt.yticks(np.arange(0, 101, 10))

# Show the legend
plt.legend()

# Store the figure
# Figures are saved twice: in .png format (for human viewing) and .eps format (to insert them into LaTeX)
plt.savefig('lineas.png' if title is None else 'lineas-'+title+'.png', bbox_inches='tight', dpi=1200)
plt.savefig('lineas.eps' if title is None else 'lineas-'+title+'.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Lines plot stored")

# SCORE:

# Create the figure
plt.figure(figsize=(12,8))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_scores) in zip(epochs, scores):
    plt.plot(internal_epochs, internal_scores, label=legend_name)


# Add the title and the axes title
plt.title('PUNTUACIÓN OBTENIDA' if title is None else ('PUNTUACIÓN OBTENIDA (' + title + ')'))
plt.xlabel('Epochs realizados')
plt.ylabel('Puntuación obtenida')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Show the legend
plt.legend()

# Store the figure
plt.savefig('puntuacion.png' if title is None else 'puntuacion-'+title+'.png', bbox_inches='tight', dpi=1200)
plt.savefig('puntuacion.eps' if title is None else 'puntuacion-'+title+'.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Scores plot stored")

# ACTIONS TAKEN

# Create the figure
plt.figure(figsize=(12,8))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_actions) in zip(epochs, actions):
    plt.plot(internal_epochs, internal_actions, label=legend_name)


# Add the title and the axes title
plt.title('ACCIONES REALIZADAS' if title is None else ('ACCIONES REALIZADAS (' + title + ')'))
plt.xlabel('Epochs realizados')
plt.ylabel('Acciones realizadas')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Show the legend
plt.legend()

# Store the figure
plt.savefig('acciones.png' if title is None else 'acciones-'+title+'.png', bbox_inches='tight', dpi=1200)
plt.savefig('acciones.eps' if title is None else 'acciones-'+title+'.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Actions taken plot stored")

# Display all figures
plt.show()
