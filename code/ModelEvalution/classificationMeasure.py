import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, recall_score


## traditional non-effort-aware performance measures
def evaluateMeasure(y_true, y_pred):
    # pre<0.5  =0；  pre>=0.5  =1；
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if fp + tn != 0:
        pf = fp / (fp + tn)
    else:
        pf = 0

    if recall + precision != 0:
        F1 = 2 * recall * precision / (recall + precision)
    else:
        F1 = 0

    AUC = roc_auc_score(y_true, y_pred)

    temp = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
    if temp != 0:
        MCC = (tp * tn - fn * fp) / np.sqrt(temp)
    else:
        MCC = 0

    if recall + 1 - pf != 0:
        g_measure = 2 * recall * (1 - pf) / (recall + 1 - pf)
    else:
        g_measure = 0

    # g_mean = np.sqrt(recall * (1 - pf))
    #
    # bal = 1 - (np.sqrt((0 - pf) ** 2 + (1 - recall) ** 2) / np.sqrt(2))

    # return [precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC]
    return [AUC, MCC, precision, recall, F1]
    # return [AUC, MCC]
