import numpy as np
from numpy import int64


# Threshold Clustering Labeling Method
def TCL(data):
    # number of samples * number of features
    [N, M] = data.shape

    # log transform
    log_data = data + 1
    log_data = np.log(log_data)

    # calculate the threshold of each metric
    mu = np.zeros(M)
    omega = np.zeros(M)
    tm = np.zeros(M)
    for i in range(M):
        mu[i] = np.mean(log_data[:, i])
        omega[i] = np.std(log_data[:, i])
        tm[i] = mu[i] + omega[i]

    original_data = np.exp(log_data)
    original_tm = np.exp(tm)

    # calculate the cluster vector
    Z = np.zeros(N)
    for i in range(N):
        t = 0
        for j in range(M):
            if original_data[i, j] > original_tm[j]:
                t = t+1
        Z[i] = t

    # label the dataset based on the values of Z
    max_Z = np.floor(np.max(Z) / 2)

    # preLabel = np.zeros(N)
    # for i in range(N):
    #     if Z[i] > max_Z:
    #         preLabel[i] = 1
    #     else:
    #         preLabel[i] = 0

    preLabel = Z > max_Z
    preLabel = preLabel.astype(int64)

    return preLabel

