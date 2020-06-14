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
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

# FILTER VALUES #

# Smooth factor (a bigger N means a smoother curve)
n = 20
b = [1.0 / n] * n
a = 1


# AUXILIAR METHODS #

def roundup_to_tens(number):
    """Rounds a number to the tens"""
    return int(math.ceil(number / 10.0)) * 10

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

# Parse the arguments
arguments = vars(parser.parse_args())

files_loaded = arguments['file']
if len(files_loaded) <= 0:
    print("ERROR: At least one file needs to be loaded.")
    sys.exit()

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
plt.figure(figsize=(10, 6))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_lines) in zip(epochs, lines):
    plt.plot(internal_epochs, internal_lines, label=legend_name)

# Add the axes title
plt.xlabel('Epochs realizados')
plt.ylabel('Lineas eliminadas')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)
# Fix the number of Y values
plt.yticks(np.arange(0, 101, 10))

# Show the legend
plt.legend()

# Store the figure
# Figures are saved twice: in .png format (for human viewing) and .eps format (to insert them into LaTeX)
plt.savefig('lineas.png', bbox_inches='tight', dpi=1200)
plt.savefig('lineas.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Lines plot stored")

# TOTAL LINES
# This plot is created since the original plot is mostly useless with how few lines there are

# Create the figure
plt.figure(figsize=(12, 6))

# Prepare the groups and bars
names = []
values = []

for agent in lines:
    names.append(agent[0])
    values.append(sum(agent[1]))

# Create the bar plot
plt.bar(names, values)

# Add the axes titles
plt.ylabel('Lineas eliminadas (total)')
# Fix the Y axis (use only ints)
max_val = roundup_to_tens(max(values)) + 10
plt.yticks(range(0, max_val, 10))

# Store the figure
plt.savefig('lineas_totales.png', bbox_inches='tight', dpi=1200)
plt.savefig('lineas_totales.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Total lines plot stored")

# SCORE (UNSMOOTHED):

# Create the figure
plt.figure(figsize=(10, 6))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_scores) in zip(epochs, scores):
    plt.plot(internal_epochs, internal_scores, label=legend_name)

# Show the legend
plt.legend()

# Add the axes title
plt.xlabel('Epochs realizados')
plt.ylabel('Puntuación obtenida')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Store the figure
plt.savefig('puntuacion.png', bbox_inches='tight', dpi=1200)
plt.savefig('puntuacion.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Scores plot (unsmoothed) stored")

# SCORE (SMOOTHED):

# Create the figure
plt.figure(figsize=(10, 6))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_scores) in zip(epochs, scores):
    yy = lfilter(b, a, internal_scores)
    plt.plot(internal_epochs, yy, label=legend_name)

# Show the legend
plt.legend()

# Add the axes title
plt.xlabel('Epochs realizados')
plt.ylabel('Puntuación obtenida')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Store the figure
plt.savefig('puntuacion_smooth.png', bbox_inches='tight', dpi=1200)
plt.savefig('puntuacion_smooth.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Scores plot (smoothed) stored")

# ACTIONS TAKEN (UNSMOOTHED)

# Create the figure
plt.figure(figsize=(10, 6))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_actions) in zip(epochs, actions):
    plt.plot(internal_epochs, internal_actions, label=legend_name)


# Add the title and the axes title
plt.xlabel('Epochs realizados')
plt.ylabel('Acciones realizadas')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Show the legend
plt.legend()

# Store the figure
plt.savefig('acciones.png', bbox_inches='tight', dpi=1200)
plt.savefig('acciones.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Actions taken plot (unsmoothed) stored")

# ACTIONS TAKEN (SMOOTHED)

# Create the figure
plt.figure(figsize=(10, 6))

# Zip and start adding lines
for (legend_name, internal_epochs), (_, internal_actions) in zip(epochs, actions):
    yy = lfilter(b, a, internal_actions)
    plt.plot(internal_epochs, yy, label=legend_name)


# Add the title and the axes title
plt.xlabel('Epochs realizados')
plt.ylabel('Acciones realizadas')

# Specify the number of ticks for the X axis (11 ticks)
plt.locator_params(axis='x', nbins=11)

# Fix the Y axis to 0
plt.ylim(bottom=0)

# Show the legend
plt.legend()

# Store the figure
plt.savefig('acciones_smooth.png', bbox_inches='tight', dpi=1200)
plt.savefig('acciones_smooth.eps', bbox_inches='tight', format='eps', dpi=1200)
print("Actions taken plot (smoothed) stored")

# Print relevant info about all agents
# Total and mean lines
total_lines = []
mean_lines = []

for line in lines:
    total = sum(line[1])
    total_lines.append((line[0], total))
    mean_lines.append((line[0], total / len(line[1])))

print("TOTAL LINES:")
for t in total_lines:
    print(t[0] + ": " + str(t[1]))
print("MEAN LINES PER EPOCH:")
for m in mean_lines:
    print(m[0] + ": " + str(m[1]))

# Mean score
mean_scores = []

for score in scores:
    total = sum(score[1])
    mean_scores.append((score[0], total / len(score[1])))

print("MEAN SCORE PER EPOCH:")
for m in mean_scores:
    print(m[0] + ": " + str(m[1]))

# Mean actions
mean_actions = []

for action in actions:
    total = sum(action[1])
    mean_actions.append((action[0], total / len(action[1])))

print("MEAN ACTIONS PER EPOCH:")
for m in mean_actions:
    print(m[0] + ": " + str(m[1]))
# Display all figures
plt.show()
