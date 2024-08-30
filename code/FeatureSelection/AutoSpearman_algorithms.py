import pandas as pd
import numpy as np
from FeatureSelectionAlgorithms import AutoSpearman

def getSub(train_data, test_data):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()

    test_data_AutoSpearman = pd.DataFrame(TEST_DATA)
    result = AutoSpearman.AutoSpearman(test_data_AutoSpearman)
    feaSubsets = result.columns.values
    feaSubsets = feaSubsets.astype(np.int)

    order = [i + 1 for i in feaSubsets]

    test_data_sub_AutoSpearman = result.values
    train_data_sub_AutoSpearman =  TRAIN_DATA[:, feaSubsets]

    return train_data_sub_AutoSpearman, test_data_sub_AutoSpearman, np.array(order).reshape((1, -1))