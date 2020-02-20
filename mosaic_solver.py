"""
Solve a given mosaic riddle (as .csv file), by treating it as a system of equations.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.optimize import lsq_linear


def main():

    if len(sys.argv) != 2:
        print('Error: No mosaic file given!')
        print('Usage: {} RIDDLE_FILE'.format(sys.argv[0]))
        sys.exit()

    riddle_file = sys.argv[1]
    riddle = read_riddle_from_file(riddle_file)

    A, b = construct_equations(riddle)
    a = np.round(lsq_linear(A, b, (0, 1)).x)

    solved_riddle = a.reshape(riddle.shape).astype('int16')

    plot_solved_mosaic(riddle, solved_riddle, riddle_file)


def read_riddle_from_file(filename):

    allowed = [chr(x+48) for x in range(10)]

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        riddle = list()
        for line in reader:
            # convert to integer array
            line_conv = [int(x) if x in allowed else np.nan for x in line]
            riddle.append(line_conv)

    return np.asarray(riddle)


def construct_equations(riddle):

    n_rows, n_cols = riddle.shape

    mask = np.isnan(riddle)
    num_cells = riddle.size
    num_hints = num_cells - np.sum(mask)
    A = np.empty((num_hints, num_cells))
    b = np.empty(num_hints)

    for row, col, i in zip(*np.where(np.logical_not(mask)), range(num_hints)):
        # set appropriate cell in result vector
        b[i] = riddle[row, col]

        # generate row for system matrix
        tmp = np.zeros_like(riddle)
        r_l = max(0, row - 1)
        r_h = min(row + 2, tmp.shape[0])
        c_l = max(0, col - 1)
        c_h = min(col + 2, tmp.shape[1])
        tmp[r_l: r_h, c_l: c_h] = 1
        A[i] = tmp.reshape(-1)

    return A, b


def plot_solved_mosaic(riddle, solved_riddle, title):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    im = ax.imshow(np.logical_not(solved_riddle), cmap='gray', vmax=1, vmin=0)

    mask = np.isnan(riddle)
    for i, j in zip(*np.where(np.logical_not(mask))):
        text = ax.text(j, i, int(riddle[i, j]), ha='center', va='center', color='gray')

    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.show()


if __name__ == '__main__':
    main()
