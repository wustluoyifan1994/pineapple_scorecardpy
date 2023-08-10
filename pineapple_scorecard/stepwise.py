# -*- coding: utf-8 -*-
# @Time : 2021/8/31 16:30
# @Author : 罗一帆
# @File : stepwise.py

import pandas as pd
import warnings
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings("ignore")


def metric(y, y_proba, metric_method="auc"):
    if metric_method == "auc":
        return roc_auc_score(y_true=y, y_score=y_proba)
    elif metric_method == "ks":
        fpr, tpr, _ = roc_curve(y_true=y, y_score=y_proba)
        return np.abs(tpr - fpr).max()
    else:
        print("metric should be in ['auc'、'ks']")
        return 0


class StepWise:

    def __init__(self,
                 label="label",
                 start_features=None,
                 solver_method="lbfgs",
                 criterion="aic",
                 direction="both",
                 stepwise_iters=35,
                 maxiters=35,
                 pvalue=0.05):
        """
        逐步回归筛选变量

        Parameters
        ----------
        label : str, optional
            目标列, by default "label"
        start_features : list, optional
            初始化特征集合, by default None
        solver_method : str, optional 'bfgs'、'lbfgs'、'ncg'
            求解方法, by default "lbfgs"
        criterion : str, optional 'aic'、'bic'、'prsquared'、'auc'、'ks'
            信息准则, by default "aic"
        direction : str, optional 'forward'、'backward'、'both',建议'both'
            逐步回归方向, by default "both"
        stepwise_iters : int, optional
            逐步回归迭代次数, by default 35
        maxiters : int, optional
            逻辑回归模型最大迭代次数, by default 35
        pvalue : float, optional
            逻辑回归模型p值, by default 0.05
        """
        self.label = label
        self.start_features = start_features
        self.solver_method = solver_method
        self.criterion = criterion
        self.direction = direction
        self.stepwise_iters = stepwise_iters
        self.maxiters = maxiters
        self.pvalue = pvalue

    def forward(self,
                data,
                best_new_score,
                remaining_features,
                selected_features,
                order):
        """
        向前逐步回归

        Parameters
        ----------
        data : frame
            包含目标列的数据集
        best_new_score : float
            最佳分数
        remaining_features : list
            剩余特征集合
        selected_features : list
            已选特征集合
        order : int
            信息准则方向; 如果是'aic'、'bic', 取-1; 否则取-1

        Returns
        -------
        dict
            包含最佳分数、剩余特征集合、已选特征集合
        """
        print("向前挑选变量".center(79, "-"))
        current_score = best_new_score
        critertion_without_candidates = {}
        for candidate_feature in remaining_features:
            formula = '{} ~ {}'.format(
                self.label, ' + '.join(selected_features +
                                       [candidate_feature])
            )
            model = smf.logit(
                formula=formula, data=data
            ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)

            # 只有所有系数非负才进行criterion的比较
            coefs = model.params.drop("Intercept")
            pvalues = model.pvalues.drop("Intercept")
            # & all(pvalues <= self.pvalue)
            if all(coefs >= 0):
                if self.criterion in ['auc', 'ks']:
                    y_proba = model.predict(data)
                    critertion_without_candidates[candidate_feature] = metric(
                        data[self.label], y_proba, metric_method=self.criterion)
                else:
                    critertion_without_candidates[candidate_feature] = (
                        eval("model.%s" % self.criterion
                             ))
        candidate_score = pd.Series(critertion_without_candidates)

        if candidate_score.shape[0] > 0:
            best_new_score = candidate_score.min()
            best_candidate = candidate_score.idxmin()
            if (current_score - best_new_score) * order < 0:
                remaining_features.remove(best_candidate)
                selected_features.append(best_candidate)
                print("向前添加了 '{}' ".format(best_candidate))
                print("向前的%s的分数%f" % (self.criterion, best_new_score))
                return {
                    "best_new_score": best_new_score,
                    "remaining_features": remaining_features,
                    "selected_features": selected_features
                }

    def backford(self,
                 data,
                 best_new_score,
                 remaining_features,
                 selected_features,
                 order):
        """
        向后逐步回归

        Parameters
        ----------
        data : frame
            包含目标列的数据集
        best_new_score : float
            最佳分数
        remaining_features : list
            剩余特征集合
        selected_features : list
            已选特征集合
        order : int
            信息准则方向, 如果是'aic'、'bic', 取-1; 否则取-1

        Returns
        -------
        dict
            包含最佳分数、剩余特征集合、已选特征集合
        """
        print("向后挑选变量".center(79, "-"))
        current_score = best_new_score
        critertion_without_candidates = {}
        for candidate_feature in selected_features:
            formula = '{} ~ {}'.format(
                self.label, ' + '.join(
                    [i for i in selected_features if i != candidate_feature])
            )
            model = smf.logit(
                formula=formula, data=data
            ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)
            if self.criterion in ['auc', 'ks']:
                y_proba = model.predict(data)
                critertion_without_candidates[candidate_feature] = metric(
                    data[self.label], y_proba, metric_method=self.criterion)
            else:
                critertion_without_candidates[candidate_feature] = (
                    eval("model.%s" % self.criterion
                         ))
        candidate_score = pd.Series(critertion_without_candidates)
        if candidate_score.shape[0] > 0:
            best_new_score = candidate_score.min()
            best_candidate = candidate_score.idxmin()
            if (current_score - best_new_score) * order < 0:
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                print("向后删除了 '{}' ".format(best_candidate))
                print("向后的%s的分数%f" % (self.criterion, best_new_score))
                return {
                    "best_new_score": best_new_score,
                    "remaining_features": remaining_features,
                    "selected_features": selected_features
                }

    def init_model(self, data, order):
        """
        初始化逐步回归

        Parameters
        ----------
        data : frame
            包含目标列的数据集
        order : int
            信息准则方向; 如果是'aic'、'bic', 取-1; 否则, 取1

        Returns
        -------
        dict
            包含最佳分数、剩余特征集合、已选特征集合
        """
        init_result = {}
        # 向前、BOTH
        if self.direction in ['forward', 'both']:
            # 有初始化特征集合
            if self.start_features is not None:
                all_features = data.drop(self.label, axis=1).columns.tolist()
                init_result['remaining_features'] = [
                    i for i in all_features if i not in self.start_features]
                init_result['selected_features'] = self.start_features.copy()

                # 训练模型
                formula = "{} ~ {}".format(
                    self.label, " + ".join(self.start_features))
                model = smf.logit(
                    formula=formula, data=data
                ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)
                if self.criterion in ["auc", "ks"]:
                    y_proba = model.predict(data)
                    init_result["current_score"] = metric(
                        data[self.label], y_proba, metric_method=self.criterion)
                    init_result["best_new_score"] = metric(
                        data[self.label], y_proba, metric_method=self.criterion)
                else:
                    init_result['current_score'] = eval(
                        "model.%s" % self.criterion)
                    init_result['best_new_score'] = eval(
                        "model.%s" % self.criterion)

            # 无初始化特征集合
            else:
                init_result["selected_features"] = []
                init_result["remaining_features"] = data.drop(
                    self.label, axis=1).columns.tolist()
                init_result['current_score'] = -float('inf') * order
                init_result['best_new_score'] = -float('inf') * order
        # 向后
        else:
            init_result["selected_features"] = data.drop(
                self.label, axis=1).columns.tolist()
            init_result["remaining_features"] = []
            formula = "{} ~ {}".format(
                self.label, " + ".join(init_result['selected_features']))
            model = smf.logit(
                formula=formula, data=data
            ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)
            if self.criterion in ["auc", "ks"]:
                y_proba = model.predict(data)
                init_result["current_score"] = metric(
                    data[self.label], y_proba, metric=self.criterion)
                init_result["best_new_score"] = metric(
                    data[self.label], y_proba, metric=self.criterion)
            else:
                init_result['current_score'] = eval(
                    "model.%s" % self.criterion)
                init_result['best_new_score'] = eval(
                    "model.%s" % self.criterion)
        return init_result

    def fit(self, X, y):
        """
        逐步回归训练

        Parameters
        ----------
        X : frame
            包含特征的数据集
        y : frame or Series
            目标列

        Returns
        -------
        list
            逐步回归挑选的特征集合
        """
        data = pd.concat([X, y], axis=1)

        # 根据信息准则确定目标是使逐步回归最小还是最大，如果是aic、bic则是最小化；否贼则是最大化
        order = -1 if self.criterion in ['aic', 'bic'] else 1

        # 初始化参数
        init_result = self.init_model(data, order)
        selected_features = init_result['selected_features']
        remaining_features = init_result['remaining_features']
        # current_score = init_result['current_score']
        best_new_score = init_result['best_new_score']
        stepwise_iters = 0

        # 逐步回归
        while True:
            stepwise_iters += 1
            if stepwise_iters > self.stepwise_iters:
                break
            if len(remaining_features) > 0:
                # 向前、BOTH
                if self.direction in ['forward', 'both']:
                    forward_result = self.forward(data=data,
                                                  best_new_score=best_new_score,
                                                  remaining_features=remaining_features,
                                                  selected_features=selected_features,
                                                  order=order
                                                  )
                    # 赋值
                    if forward_result is not None:
                        best_new_score, remaining_features, selected_features = forward_result.values()
                    else:
                        print("向前没有挑选到合适的变量")
                        if self.direction == "forward":
                            break
                else:
                    forward_result = None
            else:
                forward_result = None

            if len(selected_features) <= 1:
                print("向后没有足够的特征集合")
            else:
                # 向后、BOTH
                if self.direction in ['backward', 'both']:
                    # 判断是否需要向后
                    backward_result = self.backford(data=data,
                                                    best_new_score=best_new_score,
                                                    remaining_features=remaining_features,
                                                    selected_features=selected_features,
                                                    order=order
                                                    )
                    if backward_result is not None:
                        best_new_score, remaining_features, selected_features = backward_result.values()
                    else:
                        print("后向后没有挑选到合适的变量")
                    if forward_result is None and backward_result is None:
                        break

        print(selected_features)
        # 最后检查所有系数是否为正、且P值显著
        if len(selected_features) > 1:
            formula = "{} ~ {}".format(
                self.label, " + ".join(selected_features))
            model = smf.logit(
                formula=formula, data=data
            ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)
            p_values = model.pvalues.drop("Intercept")
            coefs = model.params.drop("Intercept")
            # 对整体系数做一次检查，系数为负先删除P值较大的
            while any(p_values >= self.pvalue) or any(coefs < 0):
                p_values = p_values.drop(p_values.idxmax())
                if len(p_values) == 0:
                    break
                formula = "{} ~ {}".format(
                    self.label, " + ".join(p_values.index.tolist()))
                model = smf.logit(
                    formula=formula, data=data
                ).fit(method=self.solver_method, disp=-1, maxiters=self.maxiters)
                p_values = model.pvalues.drop("Intercept")
                coefs = model.params.drop("Intercept")
            return p_values.index.tolist()
        else:
            return selected_features


def lr_model(X, y):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(method="lbfgs", disp=-1)
    print(model.summary2())
    from sklearn.metrics import roc_curve, roc_auc_score
    y_proba = model.predict(X)
    print(roc_auc_score(y, y_proba))
    fpr, tpr, _ = roc_curve(y, y_proba)
    print((tpr - fpr).max())
    return model


if __name__ == '__main__':
    data = pd.read_csv("data/train_woe.csv")
    X = data.drop(["y"], axis=1)
    y = data["y"]
    X.columns = ["x_%d" % i for i in range(X.shape[1])]
    step_wise = StepWise(label="y", criterion="ks")
    select_features = step_wise.fit(X, y)
    print(select_features)
    lr_model(X[select_features], y)
    # from pineapple import plot_corr
    # corr_data = X[select_features].corr()
    # plot_corr(corr_data)
