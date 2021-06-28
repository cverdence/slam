# Load libraries
import numpy as np
from src.helpers import make_data
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import src.helpers as h

# Set parameters
num_landmarks = 5
N = 20
world_size = 100.0

measurement_range = 50.0
motion_noise = 2.0
measurement_noise = 2.0
distance = 20.0

data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)

# Print stats
time_step = 4

print('Example measurements: \n', data[time_step][0])
print('\n')
print('Example motion: \n', data[time_step][1])


# Initialize constraints
def initialize_constraints(N, num_landmarks, world_size):
    omega = np.zeros((2 * (N + num_landmarks), 2 * (N + num_landmarks)))
    omega[0][0] = 1
    omega[1][1] = 1

    xi = np.zeros((2 * (N + num_landmarks), 1))
    xi[0][0] = world_size / 2
    xi[1][0] = world_size / 2

    return omega, xi


N_test = 5
num_landmarks_test = 2
small_world = 10

initial_omega, initial_xi = initialize_constraints(N_test, num_landmarks_test, small_world)

# Visualize
plt.rcParams["figure.figsize"] = (10, 7)
sns.heatmap(DataFrame(initial_omega), cmap='Blues', annot=True, linewidths=.5)

plt.rcParams["figure.figsize"] = (1, 7)
sns.heatmap(DataFrame(initial_xi), cmap='Oranges', annot=True, linewidths=.5)


# SLAM
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    omega, xi = initialize_constraints(N, num_landmarks, world_size)

    for i in range(N - 1):
        measurements = data[i][0]
        motion = data[i][1]

        for measure in measurements:
            landmark, dx, dy = measure[0], measure[1], measure[2]
            xlandmark_omega = (N * 2) + (2 * landmark)
            xrobot_omega = (2 * i)

            omega[xrobot_omega, xrobot_omega] += 1 / measurement_noise
            omega[xrobot_omega, xlandmark_omega] -= 1 / measurement_noise
            omega[xlandmark_omega][xrobot_omega] -= 1 / measurement_noise
            omega[xlandmark_omega][xlandmark_omega] += 1 / measurement_noise

            xi[xrobot_omega] -= dx / measurement_noise
            xi[xlandmark_omega] += dx / measurement_noise

            ylandmark_omega = xlandmark_omega + 1
            yrobot_omega = xrobot_omega + 1

            omega[yrobot_omega][yrobot_omega] += 1 / measurement_noise
            omega[yrobot_omega][ylandmark_omega] -= 1 / measurement_noise
            omega[ylandmark_omega][yrobot_omega] -= 1 / measurement_noise
            omega[ylandmark_omega][ylandmark_omega] += 1 / measurement_noise

            # Change robot y and landmark y in xi
            xi[yrobot_omega] -= dy / measurement_noise
            xi[ylandmark_omega] += dy / measurement_noise

        mx, my = motion[0], motion[1]
        xoriginal_omega = i * 2
        xdestiny_omega = xoriginal_omega + 2

        omega[xoriginal_omega][xoriginal_omega] += 1 / motion_noise
        omega[xoriginal_omega][xdestiny_omega] -= 1 / motion_noise
        omega[xdestiny_omega][xoriginal_omega] -= 1 / motion_noise
        omega[xdestiny_omega][xdestiny_omega] += 1 / motion_noise

        xi[xoriginal_omega] -= mx / motion_noise
        xi[xdestiny_omega] += mx / motion_noise

        yoriginal_omega = xoriginal_omega + 1
        ydestiny_omega = xdestiny_omega + 1

        omega[yoriginal_omega][yoriginal_omega] += 1 / motion_noise
        omega[yoriginal_omega][ydestiny_omega] -= 1 / motion_noise
        omega[ydestiny_omega][yoriginal_omega] -= 1 / motion_noise
        omega[ydestiny_omega][ydestiny_omega] += 1 / motion_noise

        xi[yoriginal_omega] -= my / motion_noise
        xi[ydestiny_omega] += my / motion_noise

    mu = np.dot(np.linalg.inv(omega), xi)
    return mu


# Helper functions
def get_poses_landmarks(mu, N):
    poses = []
    for i in range(N):
        poses.append((mu[2 * i].item(), mu[2 * i + 1].item()))

    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2 * (N + i)].item(), mu[2 * (N + i) + 1].item()))

    return poses, landmarks


def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('[' + ', '.join('%.3f' % p for p in poses[i]) + ']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('[' + ', '.join('%.3f' % l for l in landmarks[i]) + ']')


# Run SLAM
mu = slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise)

if mu is not None:
    poses, landmarks = get_poses_landmarks(mu, N)
    print_all(poses, landmarks)

# Display the final world
plt.rcParams["figure.figsize"] = (10,10)

if 'poses' in locals():
    print('Last pose: ', poses[-1])
    h.display_world(int(world_size), poses[-1], landmarks)

# Test
test_data1 = [[[[1, 19.457599255548065, 23.8387362100849], [2, -13.195807561967236, 11.708840328458608], [3, -30.0954905279171, 15.387879242505843]], [-12.2607279422326, -15.801093326936487]], [[[2, -0.4659930049620491, 28.088559771215664], [4, -17.866382374890936, -16.384904503932]], [-12.2607279422326, -15.801093326936487]], [[[4, -6.202512900833806, -1.823403210274639]], [-12.2607279422326, -15.801093326936487]], [[[4, 7.412136480918645, 15.388585962142429]], [14.008259661173426, 14.274756084260822]], [[[4, -7.526138813444998, -0.4563942429717849]], [14.008259661173426, 14.274756084260822]], [[[2, -6.299793150150058, 29.047830407717623], [4, -21.93551130411791, -13.21956810989039]], [14.008259661173426, 14.274756084260822]], [[[1, 15.796300959032276, 30.65769689694247], [2, -18.64370821983482, 17.380022987031367]], [14.008259661173426, 14.274756084260822]], [[[1, 0.40311325410337906, 14.169429532679855], [2, -35.069349468466235, 2.4945558982439957]], [14.008259661173426, 14.274756084260822]], [[[1, -16.71340983241936, -2.777000269543834]], [-11.006096015782283, 16.699276945166858]], [[[1, -3.611096830835776, -17.954019226763958]], [-19.693482634035977, 3.488085684573048]], [[[1, 18.398273354362416, -22.705102332550947]], [-19.693482634035977, 3.488085684573048]], [[[2, 2.789312482883833, -39.73720193121324]], [12.849049222879723, -15.326510824972983]], [[[1, 21.26897046581808, -10.121029799040915], [2, -11.917698965880655, -23.17711662602097], [3, -31.81167947898398, -16.7985673023331]], [12.849049222879723, -15.326510824972983]], [[[1, 10.48157743234859, 5.692957082575485], [2, -22.31488473554935, -5.389184118551409], [3, -40.81803984305378, -2.4703329790238118]], [12.849049222879723, -15.326510824972983]], [[[0, 10.591050242096598, -39.2051798967113], [1, -3.5675572049297553, 22.849456408289125], [2, -38.39251065320351, 7.288990306029511]], [12.849049222879723, -15.326510824972983]], [[[0, -3.6225556479370766, -25.58006865235512]], [-7.8874682868419965, -18.379005523261092]], [[[0, 1.9784503557879374, -6.5025974151499]], [-7.8874682868419965, -18.379005523261092]], [[[0, 10.050665232782423, 11.026385307998742]], [-17.82919359778298, 9.062000642947142]], [[[0, 26.526838150174818, -0.22563393232425621], [4, -33.70303936886652, 2.880339841013677]], [-17.82919359778298, 9.062000642947142]]]

mu_1 = slam(test_data1, 20, 5, 100.0, 2.0, 2.0)
poses, landmarks = get_poses_landmarks(mu_1, 20)
print_all(poses, landmarks)


