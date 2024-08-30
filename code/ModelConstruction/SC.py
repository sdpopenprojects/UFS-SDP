import numpy as np
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def SC(data):
    # scaler = StandardScaler()
    # number of samples * number of features
    N = data.shape[0]

    # normalize software metrics
    # data = scaler.fit_transform(data)

    # construct weight graph
    W = np.dot(data, data.T)
    W[W <= 0] = 0
    W = W - np.diag(np.diag(W))

    Dnsqrt = np.diag(1 / np.sqrt(np.sum(W, axis=1) + np.finfo(float).eps))
    I = np.eye(N)
    # Lsym = I - np.dot(np.dot(Dnsqrt, W), Dnsqrt)
    Lsym = I - Dnsqrt @ W @ Dnsqrt
    Lsym = 0.5 * (Lsym + Lsym.T)

    # perform the eigen decomposition, svd descending by default
    # U, S, V = np.linalg.svd(Lsym)
    # S_reordering = np.argsort(S)
    # # S = S[S_reordering]
    # U = U[:, S_reordering]
    #
    # v = np.dot(Dnsqrt, U[:, 1])
    # v = v / np.sqrt(np.sum(v ** 2, axis=0))

    # cond_number = np.linalg.cond(Lsym)
    # print(Lsym.shape)

    # eigh ascending by default
    val, U = np.linalg.eigh(Lsym)
    v = Dnsqrt @ U[:, 1]
    v = v / np.sqrt(np.sum(v ** 2, axis=0))

    # divide the data set into two clusters
    preLabel = (v > 0)

    # label the defective and non-defective clusters
    rs = np.sum(data, axis=1)
    # print(len(rs[v > 0]))
    # print(len(rs[v < 0]))
    # print("np.mean(rs[v > 0])", np.mean(rs[v > 0]))
    # print("np.mean(rs[v < 0])", np.mean(rs[v < 0]))
    if len(rs[v < 0]) == 0:
        preLabel = (v < 0)
    elif len(rs[v > 0]) == 0:
        preLabel = (v > 0)
    elif np.mean(rs[v > 0]) < np.mean(rs[v < 0]):
        preLabel = (v < 0)

    clus_label = preLabel.astype(np.int)

    return clus_label