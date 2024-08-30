from skfeature.function.similarity_based import lap_score
from skfeature.utility.construct_W import construct_W
import numpy as np

def getSub(train_data, test_data, FeaNumCandi):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()

    W = construct_W(TEST_DATA)
    score = lap_score.lap_score(TEST_DATA, W=W)
    feaSubsets = lap_score.feature_ranking(score)

    order = [i + 1 for i in feaSubsets[0:FeaNumCandi]]

    test_data_sub_LS = TEST_DATA[:, feaSubsets[0:FeaNumCandi]]
    train_data_sub_LS = TRAIN_DATA[:, feaSubsets[0:FeaNumCandi]]

    return train_data_sub_LS, test_data_sub_LS, np.array(order).reshape((1, -1))