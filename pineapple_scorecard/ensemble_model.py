# !/usr/bin/env python
# coding=utf-8

# @Time    : 2023/7/10 10:31
# @Author  : pineapple
# @File    : ensemble_model
# @Software: PyCharm

from utils import get_auc_ks
from hyperopt import STATUS_OK, Trials, fmin
# , hp, tpe
from sklearn2pmml import PMMLPipeline, sklearn2pmml
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier


def model_mapper(model_name):
    model_dict = {
        "lgb": LGBMClassifier,
        "gbdt": GradientBoostingClassifier,
        "xgb": XGBClassifier
    }
    return model_dict[model_name]


def error_mapper(error_name):
    error_dict = {
        "balance": "train_test_ks_diff_weight * np.abs(train_ks - test_ks) / train_ks - train_ks",
        "oot": "-oot_ks",
        "test": "-test_ks"
    }
    return error_dict[error_name]


def obj(params):
    """
    自定义评价函数
    :param params:
    :return:
    """
    params = space_transform(params)
    X_train = params.pop("X_train")
    y_train = params.pop("y_train")
    X_test = params.pop("X_test")
    y_test = params.pop("y_test")
    X_oot = params.pop("X_oot")
    y_oot = params.pop("y_oot")
    n_estimators = int(params.pop("n_estimators"))
    model_name = params.pop("model_name")
    error_name = params.pop("error_name")

    train_test_ks_diff_weight = params.pop("train_test_ks_diff_weight")

    model = model_mapper(model_name)(n_estimators=n_estimators, **params)
    model.fit(X_train, y_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    oot_proba = model.predict_proba(X_oot)[:, 1]

    train_auc, train_ks = get_auc_ks(
        y_train, train_proba, is_print=False).values()
    test_auc, test_ks = get_auc_ks(y_test, test_proba, is_print=False).values()
    oot_auc, oot_ks = get_auc_ks(y_oot, oot_proba, is_print=False).values()

    error = eval(error_mapper(error_name))

    params["n_estimators"] = n_estimators
    return {
        "loss": error,
        "status": STATUS_OK,
        "params": params,
        "train_auc": train_auc,
        "train_ks": train_ks,
        "test_auc": test_auc,
        "test_ks": test_ks,
        "oot_auc": oot_auc,
        "oot_ks": oot_ks,
    }


def space_transform(params):
    """参数空间转化"""
    params["max_depth"] = params["max_depth"] + 2
    return params


def get_trials_result(trials):
    """
    获取每次迭代的记录
    :param trials:
    :return:
    """
    result = []
    for i in trials:
        result.append(i["result"])
    return (
        pd.DataFrame(result)
        .assign(
            train_test_ks_diff=lambda x: np.abs(x.train_ks - x.test_ks),
            test_oot_ks_diff=lambda x: np.abs(x.test_ks - x.oot_ks),
        )
        .sort_values(by=["test_oot_ks_diff", "train_test_ks_diff"])
        .reset_index()
        .rename(columns={"index": "evals"})
    )


def select_tree_model_best_params(params, fn, algo, max_evals, rstate=None):
    """
    xgboost 调参
    :param params: dict 参数字典
    :param fn: 自定义评价函数
    :param algo: 调参方法
    :param max_evals: 最大迭代次数
    :param rstate: 每次迭代调参时的随机种子
    :return:
    """
    trials = Trials()
    best = fmin(
        fn=fn,
        space=params,
        algo=algo,
        max_evals=max_evals,
        trials=trials,
        rstate=rstate,
    )
    best = space_transform(best)
    best["n_estimators"] = int(best["n_estimators"])
    best["random_state"] = 1
    trials_result = get_trials_result(trials)
    return {"best_prams": best, "trials_result": trials_result}


def xgboost_pipeline(params, X, y):
    """
    将xgboost放入pipeline中训练，方便保存pmml
    :param params:
    :param X:
    :param y:
    :return:
    """
    params["n_estimators"] = int(params["n_estimators"])
    pipeline = PMMLPipeline(
        steps=[
            ("xgboost",
             XGBClassifier(
                 **params))])
    pipeline.fit(X, y)
    return pipeline


def pipeline2pmml(pipeline, pmml):
    """
    保存pmml文件
    :param pipeline:
    :param pmml:
    :return:
    """
    sklearn2pmml(pipeline=pipeline, pmml=pmml)
