# !/usr/bin/env python
# coding=utf-8

# @Time    : 2023/7/10 10:29
# @Author  : pineapple
# @File    : scorecard
# @Software: PyCharm

from utils import get_auc_ks, Proba2Score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd


def lr_model(dev_woe, oot_woe, need_features, label, random_state=123, is_print=False):

    X = dev_woe[need_features]
    X = sm.add_constant(X)
    y = dev_woe[label]

    X_oot = oot_woe[need_features]
    X_oot = sm.add_constant(X_oot)
    y_oot = oot_woe[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y, test_size=0.3)
    model = sm.Logit(y_train, X_train).fit(method="lbfgs", disp=-1)
    print(model.summary2())
    train_proba = model.predict(X_train)
    test_proba = model.predict(X_test)
    oot_proba = model.predict(X_oot)

    proba2score = Proba2Score(point=600, odds=0.04, pdo=100)
    score_data = pd.DataFrame()
    ks_collection = []
    for tmp in [
        ("train", y_train, train_proba),
        ("test", y_test, test_proba),
        ("oot", y_oot, oot_proba)
    ]:
        auc, ks = get_auc_ks(tmp[1], tmp[2], is_print=is_print).values()
        ks_collection.append({
            "sample_type": tmp[0],
            "cnt": tmp[1].shape[0],
            "bad": tmp[1].sum(),
            "bad_rate": tmp[1].mean(),
            "auc": auc,
            "ks": ks
        })
        score_data = pd.concat(
            [
                score_data, pd.DataFrame({
                    "sample_type": tmp[0],
                    "y": tmp[1],
                    "score": proba2score.proba2score(tmp[2])
                })
            ]
        )
    my_type = pd.CategoricalDtype(
        categories=["train", "test", "oot"],
        ordered=True
    )
    score_data["sample_type"] = score_data["sample_type"].astype(my_type)

    return {
        "model": model,
        "ks_data": pd.DataFrame(ks_collection),
        "score_data": score_data
    }


class ModelReport:

    def __init__(self):
        pass

    def fit(self):
        # 逾期率随时间的分布
        # 逾期率随样本集合的分布
        # 缺失值的分布
        # 缺失值随时间的分布
        # 模型效果
        # 入模特征效果IV、PSI
        # 入模特征的相关性
        # 入模特征的分箱逾期率
        # cut-off
        pass
