import numpy as np
from sklearn.utils import resample  # for Bootstrap sampling


# out of sample bootstrap cross validation
def outofsample_bootstrap(data, randseed, TYPE='All'):
    # sampling with replacement, whichever is not used in training data will be used in test data
    indexs = list(data.index)
    train_idx = resample(indexs, n_samples=len(indexs), random_state=randseed)
    print("train_idx", train_idx)
    train_data = data.iloc[train_idx, :]

    # picking rest of the data not considered in training data
    test_idx = list(set(indexs) - set(train_idx))
    test_data = data.iloc[test_idx, :]

    if TYPE == 'All':
        train_label = train_data.iloc[:, -1]
        train_data = train_data.iloc[:, :-1]

        test_label = test_data.iloc[:, -1]
        test_data = test_data.iloc[:, :-1]

        # train_label[train_label > 1] = 1
        # test_label[test_label > 1] = 1

        return train_data, train_label, test_data, test_label
    else:
        return train_data, test_data
