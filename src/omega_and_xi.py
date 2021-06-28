import numpy as np

omega = np.array([[1,0,0],
                  [-1,1,0],
                  [0,-1,1]])

xi = np.array([[-3],
               [5],
               [3]])

omega_inv = np.linalg.inv(np.matrix(omega))
mu = omega_inv*xi

print(mu)