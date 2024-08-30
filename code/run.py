import numpy as np
import pandas as pd
import math
from ModelConstruction import partition_dataset
from FeatureSelection import AutoSpearman_algorithms, CFS_algorithms, LS_algorithms, MCFS_algorithms, \
    REFS_algorithms, RFF_algorithms, SPEC_algorithms, XGBF_algorithms, NONE_algorithms, \
    UFS_algorithms
import datetime
import os
import time
from ModelConstruction import ConstructModel
from sklearn.preprocessing import StandardScaler

def save_selected_feature(fs_name, selected_idx):
    # 保存索引
    path_idx = os.path.join(results_dir, 'RQ1') + '/' + dataset_name + '/' + project_name[0:len(project_name) - 4] + '/'
    path = path_idx + fs_name + '.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as file:
        print(selected_idx.astype(int))
        np.savetxt(file, selected_idx, delimiter=',', fmt='%d')

def save_evalution_measure(evalution_measure_CLA, evalution_measure_TCL, evalution_measure_sc, evalution_measure_KMedoids, evalution_measure):
    path3 = os.path.join(results_dir, 'RQ2')
    classifier_names = ['CLA', 'TCL', 'SC', 'KMedoids']
    paths = []
    for s in classifier_names:
        paths.append(os.path.join(path3, s))
    print(paths)

    performance = [evalution_measure_CLA, evalution_measure_TCL, evalution_measure_sc, evalution_measure_KMedoids]

    for t in range(len(evalution_measure)):
        for p in range(len(performance)):
            save_path = os.path.join(paths[p], evalution_measure[t])
            print("save_path", save_path)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, classifier_names[p] + '_' + evalution_measure[t] + project_name), 'a') as file:
                np.savetxt(file, performance[p][t, :].reshape((1, -1)), delimiter=',')


def basic_experiment():
    verification_num = 100
    feature_lgorithm_num = 25
    feature_num = project_data.shape[1] - 1
    print(feature_num)
    FeaNumCandi = math.ceil(float(feature_num) * 0.15)
    print("FeaNumCandi", FeaNumCandi)

    evalution_measure = ['AUC', 'MCC', 'Precision', 'Recall', 'F1', 'Popt', 'Recall@20%', 'Precision@20%', 'F1measure@20%', 'PMI', 'IFA', 'Pofb']
    evalution_measure_CLA = np.zeros((len(evalution_measure), feature_lgorithm_num))
    evalution_measure_TCL = np.zeros((len(evalution_measure), feature_lgorithm_num))
    evalution_measure_sc = np.zeros((len(evalution_measure), feature_lgorithm_num))
    evalution_measure_KMedoids = np.zeros((len(evalution_measure), feature_lgorithm_num))
    # evalution_measure_KMeans = np.zeros((len(evalution_measure), feature_lgorithm_num))

    for i in range(verification_num):
        print("Project dataset size：", project_data.shape)
        print("%dth validation"%(i+1))
        print(datetime.datetime.now())

        train_data, train_label, test_data, test_label, test_bug = partition_dataset.partition(project_data, i, dataset_name)
        # print("train_label", train_label)
        # print("test_label", test_label)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.fit_transform(test_data)

        if dataset_name == 'JIRA':
            #JIRA dataset, column 28,LOC,CountLineCode
            LOC = test_data[:,27]
        elif dataset_name == 'AEEEM':
            # AEEEMdataset, column 26,LOC,ck_oo_numberOfLinesOfCode
            LOC = test_data['ck_oo_numberOfLinesOfCode']
        elif dataset_name == 'PROMISE':
            LOC = test_data['loc']

        testLOC = LOC
        # train_data = train_data.values
        # test_data = test_data.values

        #Use 3 supervised feature selection algorithms for the training set (save the corresponding indexes), then select the corresponding test subset on the test set
        #Use 21 unsupervised feature selection algorithms on the test set only (save the corresponding indexes), and then select the corresponding training subset on the training set
        FS_algorithms = {'UFSoL':UFS_algorithms, 'UDFS':UFS_algorithms, 'U2FS':UFS_algorithms, 'SRCFS':UFS_algorithms, 'SOGFS':UFS_algorithms,
                         'SOCFS':UFS_algorithms, 'RUFS':UFS_algorithms, 'NDFS':UFS_algorithms, 'LLCFS':UFS_algorithms, 'Inf_FS2020':UFS_algorithms,
                         'Inf_FS':UFS_algorithms, 'FSASL':UFS_algorithms, 'FMIUFS':UFS_algorithms, 'CNAFS':UFS_algorithms, 'jelsr_lle':UFS_algorithms,
                         'GLSPFS':UFS_algorithms, 'LS':LS_algorithms, 'MCFS':MCFS_algorithms, 'NONE':NONE_algorithms, 'REFS':REFS_algorithms,
                         'AutoSpearman':AutoSpearman_algorithms, 'SPEC':SPEC_algorithms, 'CFS':CFS_algorithms, 'XGBF':XGBF_algorithms, 'RFF':RFF_algorithms, }

        # Processing each feature selection algorithm
        FS_name = ['NONE', 'LS', 'MCFS', 'REFS', 'AutoSpearman', 'SPEC', 'CFS', 'XGBF', 'RFF', 'UFSoL', 'UDFS', 'U2FS', 'SRCFS', 'SOGFS', 'SOCFS', 'RUFS', 'NDFS', 'LLCFS', 'Inf_FS2020', 'Inf_FS',
                'FSASL', 'FMIUFS', 'CNAFS', 'jelsr_lle', 'GLSPFS']
        # FS_name = ['LS', 'MCFS', 'REFS', 'AutoSpearman', 'SPEC', 'CFS', 'XGBF', 'RFF']

        fs_number = 0
        for fs_name in FS_name:
            print(fs_name)
            algorithm = FS_algorithms[fs_name]
            if fs_name in ['CFS', 'XGBF', 'RFF']:
                train_data_sub, test_data_sub, selected_idx = algorithm.getSub(train_data, test_data, train_label, dataset_name)
            elif fs_name in ['LS', 'MCFS', 'REFS', 'SPEC']:
                train_data_sub, test_data_sub, selected_idx = algorithm.getSub(train_data, test_data, FeaNumCandi)
            elif fs_name in ['AutoSpearman']:
                train_data_sub, test_data_sub, selected_idx = algorithm.getSub(train_data, test_data)
            elif fs_name == 'NONE':
                print("NONE")
                train_data_sub, test_data_sub, selected_idx = algorithm.getSub(train_data, test_data, dataset_name)
            else:
                train_data_sub, test_data_sub, selected_idx = algorithm.getSub(train_data, test_data, fs_name, current_dir)
            print(fs_name, 'selected_idx', selected_idx, type(selected_idx))
            selected_idx = np.reshape(selected_idx, (1,-1))

            # Saves an index of the selected features
            if fs_name != 'NONE':
                save_selected_feature(fs_name, selected_idx)

            #Training the model with train_data_sub, testing with test_data_sub
            my_classifier = ConstructModel.Classifiers(test_label, test_data_sub, fs_number, testLOC, project_name, fs_name, selected_idx,
                                                       feature_num, dataset_name, results_dir, test_bug)

            evalution_measure_CLA = my_classifier.CLA_classifier(evalution_measure_CLA)
            evalution_measure_TCL = my_classifier.TCL_classifier(evalution_measure_TCL)
            evalution_measure_sc = my_classifier.SC_classifier(evalution_measure_sc)
            evalution_measure_KMedoids = my_classifier.KMedoids_classifier(evalution_measure_KMedoids)
            # evalution_measure_KMeans = my_classifier.KMeans_classifier(evalution_measure_KMeans)

            fs_number = fs_number + 1

        save_evalution_measure(evalution_measure_CLA, evalution_measure_TCL, evalution_measure_sc, evalution_measure_KMedoids, evalution_measure)

if __name__ == '__main__':
    time1 = time.time()
    current_dir = os.getcwd()

    parent_dir = os.path.dirname(current_dir)

    Dataset_Dir = os.path.join(os.path.join(parent_dir, 'dataset'))
    Dataset_List = os.listdir(Dataset_Dir)
    results_dir = os.path.join(parent_dir, 'basic_experiment_results')

    for Dataset in Dataset_List:
        dataset_dir = os.path.join(Dataset_Dir, Dataset)
        print(dataset_dir)

        file_list = os.listdir(dataset_dir)
        project_list = [file_name for file_name in file_list if file_name.endswith(".csv")]
        print("find project file : ", project_list)

        if 'JIRA' in dataset_dir:
            dataset_name = 'JIRA'
        elif 'AEEEM' in dataset_dir:
            dataset_name = 'AEEEM'
        elif 'PROMISE' in dataset_dir:
            dataset_name = 'PROMISE'

        k = 0
        for project_name in project_list:
            print("%d dataset is being processed" % (k + 1), project_name[0:len(project_name) - 4])
            file_path = os.path.join(dataset_dir, project_name)
            project_data = pd.read_csv(file_path)

            '''Repeat the out-of-sample bootstrap 100 times; divide the training set and test set; perform feature selection, 
            save the index of the selected features used in RQ1, construct the model, 
            and save the feature importance scores and performance index results.'''
            basic_experiment()
            print("--------------")

            k = k + 1

    time2 = time.time()

    print("running time：", time2 - time1)