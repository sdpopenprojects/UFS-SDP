from sklearn.cluster import KMeans
from ModelConstruction import SC, CLA, TCL
from sklearn_extra.cluster import KMedoids
from ModelEvalution import classificationMeasure, rankMeasure
from ModelConstruction import ModelInterpretation

class Classifiers:
    def __init__(self, test_label, test_data, fs_number, testLOC, project_name, fs_name, selected_idx, feature_num,
                                                                dataset_name, results_dir, test_bug):
        self.test_label = test_label
        self.test_data = test_data
        self.fs_number = fs_number
        self.testLOC = testLOC
        self.project_name = project_name
        self.fs_name = fs_name
        self.selected_idx = selected_idx
        self.feature_num = feature_num
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.test_bug = test_bug

    def measure(self, predict_label, evalution_measure, model_name):
        measure_classifier = classificationMeasure.evaluateMeasure(self.test_label, predict_label)
        measure_effort = rankMeasure.rank_measure(predict_label, self.testLOC, self.test_label, self.test_bug)

        for j in range(5):
            evalution_measure[j][self.fs_number] = measure_classifier[j]
        for j in range(7):
            evalution_measure[j + 5][self.fs_number] = measure_effort[j]

        ModelInterpretation.unsupervised_model_interpretation(predict_label, self.test_data, self.test_label, model_name,
                                                              self.project_name,
                                                              self.fs_name, self.selected_idx, self.feature_num,
                                                              self.dataset_name, self.results_dir)
        return evalution_measure

    def CLA_classifier(self, evalution_measure_CLA):
        predict_label = CLA.cla(self.test_data)
        model_name = 'CLA'
        evalution_measure_CLA = self.measure( predict_label, evalution_measure_CLA, model_name)
        return evalution_measure_CLA

    def TCL_classifier(self, evalution_measure_TCL):
        predict_label = TCL.TCL(self.test_data)
        model_name = 'TCL'
        evalution_measure_TCL = self.measure(predict_label, evalution_measure_TCL, model_name)
        return evalution_measure_TCL


    def KMedoids_classifier(self, evalution_measure_KMedoids):
        model_name = 'KMedoids'
        test_data0 = self.test_data
        culster0 = KMedoids(n_clusters=2).fit_predict(test_data0) # Clustered into two clusters

        # label the defective and non-defective clusters
        predict_label = ModelInterpretation.labelCluster(test_data0, culster0)

        evalution_measure_KMedoids = self.measure(predict_label, evalution_measure_KMedoids, model_name)

        return evalution_measure_KMedoids

    def KMeans_classifier(self, evalution_measure_KMeans):
        model_name = 'KMeans'
        test_data0 = self.test_data
        culster0 = KMeans(n_clusters=2).fit_predict(test_data0) # Clustered into two clusters

        # label the defective and non-defective clusters
        predict_label = ModelInterpretation.labelCluster(test_data0, culster0)

        evalution_measure_KMeans = self.measure(predict_label, evalution_measure_KMeans, model_name)

        return evalution_measure_KMeans

    def SC_classifier(self, evalution_measure_SC):
        model_name = 'SC'
        predict_label = SC.SC(self.test_data)
        evalution_measure_SC = self.measure(predict_label, evalution_measure_SC, model_name)
        return evalution_measure_SC
