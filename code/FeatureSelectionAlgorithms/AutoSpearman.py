import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def AutoSpearman(X_train, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5):
    """An automated feature selection approach that address collinearity and multicollinearity.
    For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.
    Parameters
    ----------
    X_train : :obj:`pd.core.frame.DataFrame`
        The X_train data to be processed
    correlation_threshold : :obj:`float`
        Threshold value of correalation.
    correlation_method : :obj:`str`
        Method for solving the correlation between the features.
    VIF_threshold : :obj:`int`
        Threshold value of VIF score.
    """
    X_AS_train = X_train.copy()
    AS_metrics = X_AS_train.columns
    count = 1

    # (Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test.
    # print('(Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test')
    while True:
        corrmat = X_AS_train.corr(method=correlation_method)
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)

        # identify correlated metrics with the correlation threshold of the threshold
        highly_correlated_metrics = ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold)) & (
            corrmat != 1)
        n_correlated_metrics = np.sum(np.sum(highly_correlated_metrics))

        if n_correlated_metrics > 0:
            # find the strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index=False)
            find_top_corr.reset_index(inplace=True)
            find_top_corr = find_top_corr[find_top_corr['value'] != 1]
            top_corr_index = find_top_corr['value'].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            # get the 2 correlated metrics with the strongest correlation
            correlated_metric_1 = top_corr_i[0]
            correlated_metric_2 = top_corr_i[1]
            # print('> Step', count, 'comparing between',
            #       correlated_metric_1, 'and', correlated_metric_2)

            # compute their correlation with other metrics outside of the pair
            correlation_with_other_metrics_1 = np.mean(abs_corrmat[correlated_metric_1][
                [i for i in top_corr_features if
                 i not in [correlated_metric_1, correlated_metric_2]]])
            correlation_with_other_metrics_2 = np.mean(abs_corrmat[correlated_metric_2][
                [i for i in top_corr_features if
                 i not in [correlated_metric_1, correlated_metric_2]]])
            # print('>>', correlated_metric_1, 'has the average correlation of',
            #       np.round(correlation_with_other_metrics_1, 3), 'with other metrics')
            # print('>>', correlated_metric_2, 'has the average correlation of',
            #       np.round(correlation_with_other_metrics_2, 3), 'with other metrics')
            # select the metric that shares the least correlation outside of the pair and exclude the other
            if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1
            # print('>>', 'Exclude', exclude_metric)
            count = count + 1
            AS_metrics = list(set(AS_metrics) - set([exclude_metric]))
            X_AS_train = X_AS_train[AS_metrics]
            X_AS_train = X_AS_train.sort_index(axis=1)
        else:
            break

    # print('According to Part 1 of AutoSpearman,', AS_metrics, 'are selected.')

    # (Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis.
    # print('(Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis')

    # Prepare a dataframe for VIF
    X_AS_train = add_constant(X_AS_train)

    selected_features = X_AS_train.columns
    count = 1
    while True:
        # Calculate VIF scores
        vif_scores = pd.DataFrame([variance_inflation_factor(np.array(X_AS_train.values, dtype=float), i)
                                   for i in range(X_AS_train.shape[1])],
                                  index=X_AS_train.columns)
        # Prepare a final dataframe of VIF scores
        vif_scores.reset_index(inplace=True)
        vif_scores.columns = ['Feature', 'VIFscore']
        vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
        vif_scores.sort_values(by=['VIFscore'], ascending=False, inplace=True, kind='mergesort')
        print("vif_scores\n", vif_scores)

        # Find features that have their VIF scores of above the threshold
        filtered_vif_scores = vif_scores[vif_scores['VIFscore']
                                         >= VIF_threshold]

        # Terminate when there is no features with the VIF scores of above the threshold
        if len(filtered_vif_scores) == 0:
            break

        # exclude the metric with the highest VIF score
        metric_to_exclude = list(filtered_vif_scores['Feature'].head(1))[0]

        print('> Step', count, '- exclude', str(metric_to_exclude))
        count = count + 1

        selected_features = list(
            set(selected_features) - set([metric_to_exclude]))

        X_AS_train = X_AS_train.loc[:, selected_features]

    # print('Finally, according to Part 2 of AutoSpearman,',
    #       X_AS_train.columns, 'are selected.')
    X_AS_train = X_AS_train.drop('const', axis=1)
    X_AS_train = X_AS_train.sort_index(axis=1)
    return X_AS_train