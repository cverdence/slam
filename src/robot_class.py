from math import *
import random


class robot:

    def __init__(self, world_size=100.0, measurement_range=30.0,
                 motion_noise=1.0, measurement_noise=1.0):
        self.measurement_noise = 0.0
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.x = world_size / 2.0
        self.y = world_size / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0

    def rand(self):
        return random.random() * 2.0 - 1.0

    def move(self, dx, dy):

        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise

        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        else:
            self.x = x
            self.y = y
            return True

    def sense(self):

        measurements = []

        for i, landmark in enumerate(self.landmarks):
            dx = abs(landmark[0] - self.x) + self.rand() * self.measurement_noise
            dy = abs(landmark[1] - self.y) + self.rand() * self.measurement_noise

            if dx < self.measurement_range or dy < self.measurement_range:
                measurements.append([i, dx, dy])

        return measurements

    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for i in range(num_landmarks):
            self.landmarks.append([round(random.random() * self.world_size),
                                   round(random.random() * self.world_size)])
        self.num_landmarks = num_landmarks

    def __repr__(self):
        return 'Robot: [x=%.5f y=%.5f]' % (self.x, self.y)
