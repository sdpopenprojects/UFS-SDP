import numpy as np
import math

def cla(X):
    # X = pd.read_csv(r'C:\Users\86178\Desktop\dataset/ant-1.3.csv')
    # # print(X)
    n,dim = X.shape
    # print(n, dim)
    # X=X.values

    threshold = np.median(X, axis=0)
    # print(len(threshold))

    index = np.zeros((n,dim))
    for i in range(n):
        idx = X[i,:] > threshold
        index[i,idx] = 1
    # print(index)

    count = np.sum(index, 1)
    # print(count, len(count))
    unicount = list(set(count))
    # print(unicount)
    num = len(unicount)

    clusters = {key:[] for key in unicount}
    # print(clusters)
    for idxx,value in enumerate(count):
        # print(idxx,value)
        clusters[value].append(idxx)
    # print(clusters)

    k = math.ceil(num/2)

    keys = list(clusters.keys())[:k+1]
    # print(keys)
    nondefective = []
    for key in keys:
        nondefective.extend(clusters[key])
    # print(nondefective)

    keys = list(clusters.keys())[k+1:]
    # print(keys)
    defective = []
    for key in keys:
        defective.extend(clusters[key])
    # print(defective)

    preLabel = [None]*n
    for j in defective:
        preLabel[j] = 1

    for j in nondefective:
        preLabel[j] = 0

    preLabel = np.array(preLabel).astype(int)

    return preLabel

