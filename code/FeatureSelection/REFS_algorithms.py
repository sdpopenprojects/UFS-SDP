from FeatureSelectionAlgorithms import REFS
import numpy as np

def getSub(train_data, test_data, FeaNumCandi):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()

    feaSubsets = REFS.refs(TEST_DATA, FeaNumCandi)
    feaSubsets = feaSubsets.astype(np.int)

    order = [i + 1 for i in feaSubsets]

    test_data_sub_REFS = TEST_DATA[:, feaSubsets]
    train_data_sub_REFS = TRAIN_DATA[:, feaSubsets]

    return train_data_sub_REFS, test_data_sub_REFS, np.array(order).reshape((1, -1))

