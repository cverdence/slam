# Load libraries
import matplotlib.pyplot as plt
import random
import src.helpers as h


# Robot class
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


# Define a world and a robot
world_size = 10.0
measurement_range = 5.0
motion_noise = 0.2
measurement_noise = 0.2

r = robot(world_size, measurement_range, motion_noise, measurement_noise)
print(r)

# Visualize the world
plt.rcParams["figure.figsize"] = (5,5)
print(r)
h.display_world(int(world_size), [r.x, r.y])

# Movement
dx = 1
dy = 2
r.move(dx, dy)

print(r)
h.display_world(int(world_size), [r.x, r.y])

# Landmarks
num_landmarks = 3
r.make_landmarks(num_landmarks)
print(r)

h.display_world(int(world_size), [r.x, r.y], r.landmarks)
print('Landmark locations [x,y]: ', r.landmarks)

# Sense
measurements = r.sense()
print(measurements)

# Data
data = [[measurements, [dx, dy]]]
print(data)

time_step = 0
print('Measurements: ', data[time_step][0])
print('Motion: ', data[time_step][1])