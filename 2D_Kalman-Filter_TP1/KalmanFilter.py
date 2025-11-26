import numpy as np
import scipy.linalg

class Kalmanfilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u = np.array([[u_x], [u_y]])
        self.xk = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0 , dt], [0, 0 ,1 ,0] ,[0 ,0 ,0 ,1]])
        self.B = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2] ,[dt, 0] ,[0, dt]])
        self.H = np.array([[1 ,0 ,0 ,0], [0 ,1, 0 ,0]])
        self.Q = np.array([[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0], [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3], [0.5 * dt ** 3, 0, dt ** 2, 0], [0, 0.5 * dt ** 3, 0, dt ** 2]]) * std_acc ** 2
        self.R = np.array([[x_std_meas ** 2, 0], [0, y_std_meas ** 2]])
        self.P = np.eye(self.A.shape[0])

    def predict(self):
        self.xk = np.dot(self.A, self.xk) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, zk):
        # Computing the Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), scipy.linalg.inv(S))
        # Correcting the state
        self.xk = self.xk + np.dot(K, (zk - np.dot(self.H, self.xk)))
        self.P = np.dot(np.eye(self.H.shape[1]) - np.dot(K, self.H), self.P)

    