import numpy as np

def getSub(train_data, test_data, dataset_name):
    order = list(range(1,train_data.shape[1] + 1))

    return train_data, test_data, np.array(order).reshape((1, -1))