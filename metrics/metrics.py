from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scipy.stats import entropy
import scipy.spatial as ss
from scipy.special import digamma
from math import log

from sklearn.feature_selection import *
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from fairlearn.metrics import *


def barplot_metric(metric_name: str,
                   method2metrics: Dict[str, List],
                   attr_names: List[str]):
    bar_width = 0.1
    figsize = (len(attr_names) * 3, 5)

    fig = plt.subplots(figsize=figsize)
    i = 0

    # draw bars for each method
    for module_name in method2metrics:
        metrics = method2metrics[module_name]
        bar_pos = [x + i * bar_width for x in np.arange(len(metrics))]
        i += 1
        plt.bar(bar_pos, metrics, width=bar_width,
                edgecolor='grey', label=module_name)

    # set labels
    plt.suptitle(metric_name, fontweight='bold')
    methods_count = len(method2metrics)
    metrics_count = len(list(method2metrics.values())[0])
    plt.xticks([r + bar_width * ((methods_count - 1) // 2) for r in range(metrics_count)],
               attr_names)
    plt.legend()
    plt.show()


def create_cats_idx(adata, cats):
    # create numerical index for each attr in cats

    for i in range(len(cats)):
        values = list(set(adata.obs[cats[i]]))

        val_to_idx = {v: values.index(v) for v in values}

        idx_list = [val_to_idx[v] for v in adata.obs[cats[i]]]

        adata.obs[cats[i] + '_idx'] = pd.Categorical(idx_list)

    return adata


# tutorial XGBoost: https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
def clf_S_Z_metrics(adata, cats, module_name):

    print(f'Method: {module_name}')
    print(f'XGBoost classifier for Si')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    Z = [adata.obsm[f"{module_name}_Z_0"]]
    S = []

    for i in range(1, len(cats) + 1):
        Z.append(adata.obsm[f"{module_name}_Z_{i}"])
        S.append(adata.obs[cats[i - 1] + '_idx'])

    acc = []
    acc_not_concat = []
    acc_not_max = []

    acc_gap_concat = []
    acc_gap_max = []

    for i in range(1, len(cats) + 1):

        Si = S[i-1]

        acc_Si_Zj = {j: [] for j in range(len(cats) + 1)}
        acc_Si_Zi_list = []
        acc_Si_Z_not_i_list = []
        acc_Si_Z_not_i_max_list = []

        for train, test in cv.split(Z[i], Si):

            # acc (Si | Zj)
            for j in range(len(cats) + 1):
                estimator = clone(clf)
                estimator.fit(Z[j][train], Si[train], eval_set=[(Z[j][test], Si[test])], verbose=False)
                acc_Si_Zj[j].append(estimator.score(Z[j][test], Si[test]))

            # find Zi with max acc(Si | Zi)

            acc_Si_Zj_sorted_idx = sorted([j for j in acc_Si_Zj.keys()], key=lambda k: acc_Si_Zj[k][-1])
            max_idx_1 = acc_Si_Zj_sorted_idx[-1]
            max_idx_2 = acc_Si_Zj_sorted_idx[-2]
            acc_Si_Zi_list.append(acc_Si_Zj[max_idx_1][-1])
            acc_Si_Z_not_i_max_list.append(acc_Si_Zj[max_idx_2][-1])

            Z_not_i = np.concatenate([Z[j] for j in range(len(cats) + 1) if j != max_idx_1], axis=1)
            Z_not_i_train = Z_not_i[train]
            Z_not_i_test = Z_not_i[test]

            # acc (Si | Z_{-i})
            estimator = clone(clf)
            estimator.fit(Z_not_i_train, Si[train], eval_set=[(Z_not_i_test, Si[test])], verbose=False)
            acc_Si_Z_not_i_list.append(estimator.score(Z_not_i_test, Si[test]))

        acc_Si_Zi = np.mean(acc_Si_Zi_list)
        acc_Si_Z_not_i = np.mean(acc_Si_Z_not_i_list)
        acc_Si_Z_not_i_max = np.mean(acc_Si_Z_not_i_max_list)

        acc.append(acc_Si_Zi)
        acc_not_concat.append(acc_Si_Z_not_i)
        acc_not_max.append(acc_Si_Z_not_i_max)

        acc_gap_concat.append(acc_Si_Zi - acc_Si_Z_not_i)
        acc_gap_max.append(acc_Si_Zi - acc_Si_Z_not_i_max)

        print(f'acc(S_{i} | Z_{i}) = {acc_Si_Zi:.4f}, '
              f'acc(S_{i} | Z - Z_{i}) = {acc_Si_Z_not_i:.4f}, '
              f'max acc(S_{i} | Z_j!={i}) = {acc_Si_Z_not_i_max:.4f}')

    # append averages to the end of the lists

    acc.append(np.mean(acc))
    acc_not_concat.append(np.mean(acc_not_concat))
    acc_not_max.append(np.mean(acc_not_max))

    acc_gap_concat.append(np.mean(acc_gap_concat))
    acc_gap_max.append(np.mean(acc_gap_max))

    print(f'concatCAG = {acc_gap_concat[-1]:.4f}, maxCAG = {acc_gap_max[-1]:.4f}')

    return acc, acc_not_concat, acc_not_max, acc_gap_concat, acc_gap_max


def fair_clf_metrics(adata, cats, y_name, module_name):
    # fairness metrics: DP, EO, ...
    # https://fairlearn.org/v0.9/user_guide/assessment/common_fairness_metrics.html

    print(f'Method: {module_name}')

    # binarize y
    y_obs = adata.obs[y_name]
    mid = sorted(y_obs)[len(y_obs) // 2]
    y_bin = [0 if x < mid else 1 for x in adata.obs[y_name]] if mid > 0 \
        else [0 if x <= 0 else 1 for x in adata.obs[y_name]]
    y_bin_name = y_name + '_bin'
    adata.obs[y_bin_name] = pd.Categorical(y_bin)
    Y = adata.obs[y_bin_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    DP_diff = []
    DP_ratio = []
    EO_diff = []

    acc = []

    print(f'fairness metrics wrt Si for XGBoost classifier {y_bin_name} | (Z - Zi)')

    for i in range(1, len(cats) + 1):

        Z_not_i = adata.obsm[f"{module_name}_Z_not_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        DP_diff_i = []
        DP_ratio_i = []
        EO_diff_i = []

        ACC_i = []

        for train, test in cv.split(Z_not_i, Y):
            Z_not_i_train = Z_not_i[train]
            Z_not_i_test = Z_not_i[test]
            Y_train = Y[train]
            Y_test = Y[test]

            Si_test = Si[test]

            estimator = clone(clf)

            estimator.fit(Z_not_i_train, Y_train, eval_set=[(Z_not_i_test, Y_test)], verbose=False)

            Y_test = pd.Series(Y_test, dtype=int)

            Y_pred = estimator.predict(Y_test)

            dp_diff = demographic_parity_difference(Y_test, Y_pred, sensitive_features=Si_test)
            DP_diff_i.append(dp_diff)
            dp_ratio = demographic_parity_ratio(Y_test, Y_pred, sensitive_features=Si_test)
            DP_ratio_i.append(dp_ratio)
            eo_diff = equalized_odds_difference(Y_test, Y_pred, sensitive_features=Si_test)
            EO_diff_i.append(eo_diff)
            
            test_acc = estimator.score(Z_not_i_test, Y_test)
            ACC_i.append(test_acc)

        dp_diff = np.mean(DP_diff_i)
        DP_diff.append(dp_diff)
        dp_ratio = np.mean(DP_ratio_i)
        DP_ratio.append(dp_ratio)
        eo_diff = np.mean(EO_diff_i)
        EO_diff.append(eo_diff)

        test_acc = np.mean(ACC_i)
        acc.append(test_acc)

        print(f'i={i}: accuracy = {test_acc:.4f}, DP_diff = {dp_diff:.4f}, EO_diff = {eo_diff:.4f}')

    acc.append(np.mean(acc))
    DP_diff.append(np.mean(DP_diff))
    EO_diff.append(np.mean(EO_diff))

    print(f'average: accuracy = {acc[-1]:.4f}, DP_diff = {DP_diff[-1]:.4f}, EO_diff = {EO_diff[-1]:.4f}')

    return acc, DP_diff, EO_diff


def max_dim_MI_metrics(adata, cats, module_name):
    # Max Mutual Information by taking Max over Dims
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

    print(f'Method: {module_name}')
    print('Max_Dim Mutual Information metrics')

    MI_dif = []
    H = []

    MI = []
    MI_not = []

    for i in range(1, len(cats) + 1):
        Zi = adata.obsm[f"{module_name}_Z_{i}"]
        Z_not_i = adata.obsm[f"{module_name}_Z_not_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        # MI

        mi = mutual_info_classif(Zi, Si, discrete_features=False)
        mi_not = mutual_info_classif(Z_not_i, Si, discrete_features=False)

        mi_max = np.max(mi)
        mi_not_max = np.max(mi_not)

        MI_dif.append(mi_max - mi_not_max)

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

        print(f"MI(Z_{i} ; S_{i}) = {mi_max:.4f},  MI((Z - Z_{i}) ; S_{i}) = {mi_not_max:.4f}")

        MI.append(mi_max)
        MI_not.append(mi_not_max)

    # MIG

    mig = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"MIG = {mig:.4f}")

    return MI, MI_not, MI_dif, mig


def Mixed_KSG_MI_metrics(adata, cats, module_name, pre_MI=None):
    # Mutual Information by Mixed_KSG
    # code from: https://github.com/wgao9/mixed_KSG/blob/master/mixed.py
    # pre_MI: pre-calculated MI from find_Zi_by_MI_metrics (2d array)

    print(f'Method: {module_name}')
    print('Mixed_KSG Mutual Information metrics')

    Z = [adata.obsm[f"{module_name}_Z_0"]]
    Z_not = []
    S = []

    for i in range(1, len(cats) + 1):
        Z.append(adata.obsm[f"{module_name}_Z_{i}"])
        Z_not.append(adata.obsm[f"{module_name}_Z_not_{i}"])
        S.append(adata.obs[cats[i - 1] + '_idx'])

    MI = []
    MI_not = []
    MI_not_max = []

    MI_dif = []
    MI_dif_max = []
    H = []

    for i in range(1, len(cats) + 1):
        # MI

        Si = S[i-1]

        mi = Mixed_KSG_MI(Z[i], Si)
        mi_not = Mixed_KSG_MI(Z_not[i-1], Si)
        if pre_MI is None:
            mi_not_max = sorted(Mixed_KSG_MI(Z[j], Si) for j in range(len(Z)))[-2]
        else:
            mi_not_max = sorted(pre_MI[i-1][j] for j in range(len(pre_MI[i-1])))[-2]

        MI.append(mi)
        MI_not.append(mi_not)
        MI_not_max.append(mi_not_max)

        MI_dif.append(mi - mi_not)
        MI_dif_max.append(mi - mi_not_max)

        print(f"MI(Z_{i} ; S_{i}) = {mi:.4f},  "
              f"MI((Z - Z_{i}) ; S_{i}) = {mi_not:.4f}, "
              f"max MI((Z_j!={i}) ; S_{i}) = {mi_not_max:.4f}")

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

    # MIG

    maxMIG = np.mean([MI_dif_max[i] / H[i] for i in range(len(cats))])
    concatMIG = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"maxMIG = {maxMIG:.4f}, concatMIG = {concatMIG:.4f}")

    # append averages to the end of the lists
    MI.append(np.mean(MI))
    MI_not_max.append(np.mean(MI_not_max))
    MI_not.append(np.mean(MI_not))
    MI_dif_max.append(maxMIG)
    MI_dif.append(concatMIG)

    return MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG


def find_Zi_by_MI_metrics(adata, z, cats, module_name):
    z_list = np.hsplit(z, z.shape[1])
    MI = []
    max_mi_idx_list = []
    for i, c in enumerate(cats):
        Si = adata.obs[c + '_idx']
        MI_Si = []
        for j, Zj in enumerate(z_list):
            MI_Zj_Si = mutual_info_classif(Zj, Si, discrete_features=False)[0]
            MI_Si.append(MI_Zj_Si)
        max_mi_idx = max(range(len(z_list)), key=lambda k: MI_Si[k])
        max_mi_idx_list.append(max_mi_idx)
        adata.obsm[f"{module_name}_Z_{i+1}"] = z_list[max_mi_idx]
        adata.obsm[f"{module_name}_Z_not_{i + 1}"] = \
            np.concatenate(list(z_list[a] for a in range(len(z_list)) if a != max_mi_idx), axis=1)
        MI.append(MI_Si)

    adata.obsm[f"{module_name}_Z_0"] = \
        np.concatenate(list(z_list[a] for a in range(len(z_list)) if a not in max_mi_idx_list), axis=1)

    return adata, MI


def Mixed_KSG_MI(x, y, k=5):
    """
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *Mixed-KSG* mutual information estimator

        Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
        y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
    """
    x = np.array(x)
    y = np.array(y)

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))
    if y.ndim == 1:
        y = y.reshape((N, 1))
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i], 1e-15, p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i], 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], 1e-15, p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i], knn_dis[i] - 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], knn_dis[i] - 1e-15, p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans
