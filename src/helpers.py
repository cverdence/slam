from src.robot_class import robot
from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_world(world_size, position, landmarks=None):
    sns.set_style("dark")
    world_grid = np.zeros((world_size + 1, world_size + 1))

    ax = plt.gca()
    cols = world_size + 1
    rows = world_size + 1

    ax.set_xticks([x for x in range(1, cols)], minor=True)
    ax.set_yticks([y for y in range(1, rows)], minor=True)

    plt.grid(which='minor', ls='-', lw=1, color='white')
    plt.grid(which='major', ls='-', lw=2, color='white')

    ax.text(position[0], position[1], 'o', ha='center', va='center', color='r', fontsize=30)

    if landmarks is not None:
        for pos in landmarks:
            if pos != position:
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)

    plt.show()


def make_data(N, num_landmarks, world_size, measurement_range, motion_noise,
              measurement_noise, distance):

    try:
        check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise)
    except ValueError:
        print('Error: You must implement the sense function in robot_class.py.')
        return []

    complete = False

    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    while not complete:

        data = []

        seen = [False for row in range(num_landmarks)]

        orientation = random.random() * 2.0 * pi
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance

        for k in range(N - 1):
            Z = r.sense()

            for i in range(len(Z)):
                seen[Z[i][0]] = True

            while not r.move(dx, dy):
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            data.append([Z, [dx, dy]])

        complete = (sum(seen) == num_landmarks)

    print(' ')
    print('Landmarks: ', r.landmarks)
    print(r)

    return data


def check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise):
    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    test_Z = r.sense()
    if test_Z is None:
        raise ValueError
