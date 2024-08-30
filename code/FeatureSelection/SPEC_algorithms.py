from skfeature.function.similarity_based import SPEC
import numpy as np

def getSub(train_data, test_data, FeaNumCandi):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()

    W = SPEC.spec(TEST_DATA)
    feaSubsets = SPEC.feature_ranking(W)

    order = [i + 1 for i in feaSubsets[0:FeaNumCandi]]

    test_data_sub_SPEC = TEST_DATA[:, feaSubsets[0:FeaNumCandi]]
    train_data_sub_SPEC = TRAIN_DATA[:, feaSubsets[0:FeaNumCandi]]

    return train_data_sub_SPEC, test_data_sub_SPEC, np.array(order).reshape((1, -1))