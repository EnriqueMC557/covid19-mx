import numpy as np
import pandas as pd

from scipy import stats


def precision(TP, FP):
    try:
        return TP/(TP + FP)
    except ZeroDivisionError:
        return np.nan


def sensitivity(TP, FN):
    try:
        return TP/(TP + FN)
    except ZeroDivisionError:
        return np.nan


def specificity(TN, FP):
    try:
        return TN/(TN + FP)
    except ZeroDivisionError:
        return np.nan


def f1_score(TP, FP, FN):
    try:
        return (2*TP)/(2*TP + FP + FN)
    except ZeroDivisionError:
        return np.nan


def ppv(TP, FP):
    try:
        return TP/(TP + FP)
    except ZeroDivisionError:
        return np.nan


def npv(TN, FN):
    try:
        return TN/(FN + TN)
    except ZeroDivisionError:
        return np.nan


def cramers_v(x1, x2):
    crosstab = pd.crosstab(x1, x2, rownames=None, colnames=None).to_numpy()
    chi2 = stats.chi2_contingency(crosstab)[0]
    n = np.sum(crosstab)
    min_dim = min(crosstab.shape)-1
    return np.round(np.sqrt(chi2/(n*min_dim)), 2)
