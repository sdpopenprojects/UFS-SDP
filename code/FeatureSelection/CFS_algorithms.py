# from FeatureSelectionAlgorithms import CFS_
from skfeature.function.statistical_based import CFS
import numpy as np

def getSub(train_data, test_data, train_label, dataset_name):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()
    TRAIN_LABLE = train_label.copy()

    feaSubsets = CFS.cfs(TRAIN_DATA, TRAIN_LABLE)

    order = [i + 1 for i in feaSubsets]

    train_data_sub_CFS = TRAIN_DATA[:, feaSubsets]
    test_data_sub_CFS = TEST_DATA[:, feaSubsets]

    return train_data_sub_CFS, test_data_sub_CFS, np.array(order).reshape((1, -1))