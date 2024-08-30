from skfeature.function.sparse_learning_based import MCFS
import numpy as np

def getSub(train_data, test_data, FeaNumCandi):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()

    W = MCFS.mcfs(TEST_DATA, FeaNumCandi, n_clusters=2)
    feaSubsets = MCFS.feature_ranking(W)

    order = [i + 1 for i in feaSubsets[0:FeaNumCandi]]

    test_data_sub_MCFS = TEST_DATA[:, feaSubsets[0:FeaNumCandi]]
    train_data_sub_MCFS = TRAIN_DATA[:, feaSubsets[0:FeaNumCandi]]

    return train_data_sub_MCFS, test_data_sub_MCFS, np.array(order).reshape((1, -1))