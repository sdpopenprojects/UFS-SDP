from ModelConstruction import fast_bootstrapCV
import numpy as np

def partition(dataset, i, dataset_name):
    train_data, train_label, test_data, test_label = fast_bootstrapCV.outofsample_bootstrap(dataset, i)
    train_data = train_data.astype(np.float64)
    test_data = test_data.astype(np.float64)
    test_bug = test_label

    train_label = train_label.values
    test_label = test_label.values

    if dataset_name == 'JIRA':
        train_label = train_label.astype(np.float64)
        test_label = test_label.astype(np.float64)
        train_label[train_label > 0] = '1'
        train_label[train_label == 0] = '0'
        test_label[test_label > 0] = '1'
        test_label[test_label == 0] = '0'
        # train_label = train_label + 0
        # test_label = test_label + 0
    elif dataset_name == 'AEEEM':
        train_label[train_label == "buggy"] = '1'
        train_label[train_label == "clean"] = '0'
        test_label[test_label == "buggy"] = '1'
        test_label[test_label == "clean"] = '0'
    elif dataset_name == 'PROMISE':
        train_label = train_label.astype(np.float64)
        test_label = test_label.astype(np.float64)
        train_label[train_label > 0] = '1'
        train_label[train_label == 0] = '0'
        test_label[test_label > 0] = '1'
        test_label[test_label == 0] = '0'


    train_label = train_label.astype(np.int)
    test_label = test_label.astype(np.int)

    return train_data, train_label, test_data, test_label, test_bug