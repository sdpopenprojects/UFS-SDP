import warnings
from ModelConstruction.xgb_rfclassifiers import xgboost,random_forest_classifier
import pandas as pd
import numpy as np
from ModelEvalution import rankMeasure
from ModelConstruction import partition_dataset
from ModelEvalution import classificationMeasure
warnings.filterwarnings('ignore')



def forward_search(dataset_train, dataset_test, classifier_name, classifier):
    """
    apply forkward search for wrapper-based feature subset selection
    :param dataset_train:
    :param dataset_test:
    :param classifier:
    :return:
    """
    training_data_y = dataset_train.iloc[:, -1]
    training_data_x = dataset_train.iloc[:, :-1]

    testing_data_y = dataset_test.iloc[:, -1]
    testing_data_x = dataset_test.iloc[:, :-1]


    # testing_data_value = dataset_test.loc[:, 'bug'].tolist()#这里需要改成符合自己数据集的
    # testingcode = testing_data_x.iloc[:, 27].tolist()#JIRA
    # testingcode = testing_data_x.iloc[:, 25].tolist()#AEEEM
    # print("testing_data_x.shape", testing_data_x.shape)
    # print("testingcode", testingcode)

    best_AUC = 0
    column = -1

    # new_training_data_x = training_data_x
    # new_testing_data_x = testing_data_x

    for i in range(training_data_x.shape[1]):  # training_data_x的列数
        tmp_training_data_x = training_data_x[i].to_frame()
        # print("training_data_x[i].to_frame()", training_data_x[i].to_frame())
        # print("training_data_x[i].shape", training_data_x[i].shape)
        tmp_testing_data_x = testing_data_x[i].to_frame()
        # print(tmp_training_data_x.columns.values)
        new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
        predict_label = new_model.predict(tmp_testing_data_x)

        new_AUC, _, _, _, _ = classificationMeasure.evaluateMeasure(testing_data_y, predict_label)

        if new_AUC >= best_AUC:
            best_AUC = new_AUC
            column = i
            new_training_data_x = tmp_training_data_x
            new_testing_data_x = tmp_testing_data_x

    # print("column", column)

    old_column = column
    selected_cloumn = []
    dropped_column = []
    selected_cloumn.append(column)

    for i in range(training_data_x.shape[1] - 1):
        candidate_columns = []
        for col in training_data_x.columns.values.tolist():
            if col not in new_training_data_x.columns.values.tolist():
                candidate_columns.append(col)
        # print("candidate_columns", candidate_columns)
        print(new_training_data_x.columns.values.tolist())
        for j in candidate_columns:
            tmp_training_data_x = pd.concat([new_training_data_x, training_data_x[j].to_frame()], axis=1)
            tmp_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[j].to_frame()], axis=1)
            # print(tmp_training_data_x.columns.values)
            new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
            predict_label = new_model.predict(tmp_testing_data_x)

            new_AUC, _, _, _, _ = classificationMeasure.evaluateMeasure(testing_data_y, predict_label)

            if new_AUC >= best_AUC:
                best_AUC = new_AUC
                column = j

        # print("old_column", old_column)
        # print("column", column)

        if old_column != column:
            new_training_data_x = pd.concat([new_training_data_x, training_data_x[column].to_frame()], axis=1)
            new_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[column].to_frame()], axis=1)
            selected_cloumn.append(column)
            old_column = column
            if i == training_data_x.shape[1] - 2:
                for p in training_data_x.columns.values.tolist():
                    if p not in selected_cloumn:
                        dropped_column.append(p)
                # print(selected_cloumn)
                return dropped_column, selected_cloumn, best_AUC
        else:
            for k in training_data_x.columns.values.tolist():
                if k not in selected_cloumn:
                    dropped_column.append(k)
            # print(selected_cloumn)
            return dropped_column, selected_cloumn, best_AUC

def XGB_RF_ADB_feature_selection(data, classifier, dataset_name):
    classifier_name = {
        'XGB': xgboost,
        'RF': random_forest_classifier,
    }

    # 对train_FS进行交叉验证 dataset_train, dataset_test
    train_data, train_label, test_data, test_label, _ = partition_dataset.partition(data, 42, dataset_name)
    train_label = train_label.reshape((len(train_label)), 1)
    test_label = test_label.reshape((len(test_label)), 1)

    dataset_train = np.concatenate((train_data, train_label), axis=1)
    dataset_test = np.concatenate((test_data, test_label), axis=1)

    # # 训练测试都用
    dataset_train = pd.DataFrame(dataset_train)
    dataset_test = pd.DataFrame(dataset_test)
    # print(dataset_test.shape)

    print("现在运行的是：", classifier)
    _, selected_cloumn2, _ = forward_search(dataset_train, dataset_test, classifier_name, classifier)
    print("selected_cloumn2", selected_cloumn2)

    return selected_cloumn2
