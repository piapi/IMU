import numpy as np

H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


def fusion(pos_uwb, pdr_info, P, R, Q):
    pos_pdr = pdr_info[:2].T
    yaw = pdr_info[2]
    stpes_length = pdr_info[3]
    A = np.array([[1, 0, np.sin(yaw), stpes_length * np.cos(yaw)],
                  [0, 1, np.cos(yaw), -stpes_length * np.sin(yaw)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    P = A @ P @ A.T + Q
    z = pos_pdr - pos_uwb
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = K @ z
    P = (np.eye(4) - K @ H) @ P

    pos = np.array([pos_pdr[0] - x[0], pos_pdr[1] - x[1], yaw - x[3], stpes_length - x[2]])
    return pos, P, x[3], x[2]
