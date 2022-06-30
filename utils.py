import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def train_val_test_split(X, y, sizes, random_state=None, stratify=None):
    assert (sum(sizes) == 1)
    assert (len(sizes) == 3)
    test_and_vali_size = sizes[1] + sizes[2]
    rel_test_size = sizes[2] / test_and_vali_size
    if stratify is not None:
        if isinstance(stratify, pd.DataFrame) and len(stratify.columns) > 1:
            # We want to stratify by multiple columns, sklearn train_test_split does not support that.
            # Thus, set up an artificial column for that.
            # (Inspired by https://stackoverflow.com/a/51525992/2207840.)
            strat = stratify.iloc[:, 0].astype(str)
            for col_idx in range(1, len(stratify.columns)):
                strat = strat + '_' + stratify.iloc[:, col_idx].astype(str)
            assert (len(np.unique(strat)) == np.prod(stratify.nunique(axis='rows')))
        else:
            strat = stratify

        X_train, X_vt, y_train, y_vt, strat_train, strat_vt = train_test_split(X, y, strat,
                                                                               test_size=test_and_vali_size,
                                                                               random_state=random_state,
                                                                               stratify=strat)
        X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=rel_test_size, random_state=random_state,
                                                        stratify=strat_vt)
    else:
        X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=test_and_vali_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=rel_test_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def stack_tensor_datasets(td1, td2):
    return TensorDataset(*[torch.cat((td1.tensors[idx], td2.tensors[idx]), 0)
                           for idx in range(0, len(td1.tensors))])
