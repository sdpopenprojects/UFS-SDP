from FeatureSelectionAlgorithms import XGB_RF_AUC_only_train
import numpy as np
import pandas as pd

def getSub(train_data, test_data, train_label, dataset_name):
    TRAIN_DATA = train_data.copy()
    TEST_DATA = test_data.copy()
    TRAIN_LABLE = train_label.copy()

    tem_train_label = TRAIN_LABLE.reshape((len(TRAIN_LABLE)), 1)
    train_FS = np.concatenate((TRAIN_DATA, tem_train_label), axis=1)
    train_FS = pd.DataFrame(train_FS)

    feaSubsets = XGB_RF_AUC_only_train.XGB_RF_ADB_feature_selection(train_FS, 'XGB', dataset_name)

    order = [i + 1 for i in feaSubsets]

    train_data_sub_XGB = TRAIN_DATA[:, feaSubsets]
    test_data_sub_XGB = TEST_DATA[:, feaSubsets]

    return train_data_sub_XGB, test_data_sub_XGB, np.array(order).reshape((1, -1))