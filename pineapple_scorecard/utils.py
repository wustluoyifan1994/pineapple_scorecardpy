# !/usr/bin/env python
# coding=utf-8

# @Time    : 2023/7/10 10:29
# @Author  : pineapple
# @File    : utils
# @Software: PyCharm

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

from pandas import DataFrame
import scorecardpy as sc
import matplotlib.pyplot as plt
import warnings
import matplotlib
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF


warnings.filterwarnings("ignore")
matplotlib.rc("font", **{"family": "Heiti TC"})
# matplotlib.rc("font", **{"family": "SimHei})
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100
def n0(x): return sum(x == 0)


def n1(x): return sum(x == 1)


# 获取标签
def get_label(x, bad=14, good=0, default=-1):
    if pd.isnull(x):
        return np.nan
    elif x >= bad:
        return 1
    elif x <= good:
        return 0
    else:
        return default


# 概率转分数和生成评分卡
class Proba2Score:

    def __init__(self, point=600, odds=0.04, pdo=100):
        self.point = point
        self.odds = odds
        self.pdo = pdo

    def ab(self):
        b = self.pdo / np.log(2)
        a = self.point + b * np.log(self.odds)
        return {"a": a, "b": b}

    def proba2score(self, y_proba):
        a, b = self.ab().values()
        return a - b * np.log(y_proba / (1 - y_proba))

    def scorecard(self, params, bins):
        a, b = self.ab().values()
        bins_data = pd.concat(bins, ignore_index=True)
        variable = []
        bin_list = []
        score = []
        for i in params.index:
            feature = params.loc[i, "variable"]
            coef = params.loc[i, "coef"]
            if feature == "const":
                base = a - b * coef
                variable.append("Intercept")
                bin_list.append("")
                score.append(base)
            else:
                bins_tmp = bins_data.loc[bins_data["variable"]
                                         == feature.replace("_woe", "")]
                bins_tmp["score"] = bins_tmp["woe"].apply(
                    lambda x: -b * coef * x)
                variable = variable + bins_tmp["variable"].tolist()
                bin_list = bin_list + bins_tmp["bin"].tolist()
                score = score + bins_tmp["score"].tolist()

        scorecard_data = pd.DataFrame({
            "variable": variable,
            "bin": bin_list,
            "score": score
        })
        scorecard_data["score"] = scorecard_data["score"].astype(int)
        x_list = [f'x{i}' for i in range(scorecard_data["variable"].nunique())]
        mapper = dict(zip(list(scorecard_data["variable"].unique()), x_list))
        scorecard_data["variable_no"] = scorecard_data["variable"].apply(
            lambda x: mapper[x])
        return scorecard_data[[
            "variable_no", "variable", "bin", "score"
        ]]


# 多个数据框保存至excel
def dataframes_write_excel(res, file_name):
    writer = pd.ExcelWriter(file_name)
    for sheet_name, data in res.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()


def calculate_overdue_rate_by_month(
        data: DataFrame, loan_date: str, y: str) -> DataFrame:
    """
    按月计算逾期率分布
    :param data:
    :param loan_date:
    :param y:
    :return:
    """
    data = data.copy()
    data[loan_date] = data[loan_date].apply(lambda x: str(x)[:7])
    gb = data.groupby(by=loan_date)[y].agg([
        ("计数", np.size), ("坏", np.sum), ("坏样本率", np.mean)
    ])
    return gb


def calculate_missing_rate(
        data: DataFrame, y: str, features: list = None, threshold: float = 0.95) -> dict:
    """
    计算特征的缺失率
    :param data: 数据
    :param y: label
    :param features: 需要计算缺失率的特征集合
    :param threshold: 剔除特征缺失率>=N的阈值
    :return:
    """
    if features is None:
        features = data.drop(y, axis=1).columns.tolist()
    missing_data = (
        pd.DataFrame(
            {
                "variable": features,
                "missing_rate": data[features].isnull().sum() / data.shape[0],
            }
        )
        .sort_values(by="missing_rate", ascending=False)
        .reset_index(drop=True)
    )
    del_features_by_missing = missing_data.loc[
        missing_data["missing_rate"] >= threshold, "variable"
    ].tolist()
    return {
        "missing_data": missing_data,
        "del_features_by_missing": del_features_by_missing,
    }


def calculate_mode_rate(data: DataFrame, y: str,
                        features: list = None, threshold: float = 0.95) -> dict:

    if features is None:
        features = data.drop(y, axis=1).columns.tolist()
    mode_rate_list = []
    for var in features:
        mode_rate = data[var].value_counts().values[0] / data.shape[0]
        mode_rate_list.append(mode_rate)
    mode_data = (
        pd.DataFrame({"variable": features, "mode_rate": mode_rate_list})
        .sort_values(by="mode_rate", ascending=False)
        .reset_index(drop=True)
    )
    del_features_by_mode = mode_data.loc[
        mode_data["mode_rate"] >= threshold, "variable"
    ].tolist()
    return {
        "mode_data": mode_data,
        "del_features_by_mode": del_features_by_mode}


# woe分箱
def mono_woebin(data, y, x=None, special_values=None, badratediff=0):
    data = data.copy()
    if x is None:
        x = data.drop(y, axis=1).columns.tolist()
    if isinstance(x, str):
        x = [x]
    bins = {}
    for xi in x:
        try:
            for num in [5, 4, 3, 2]:
                gb = sc.woebin(
                    data,
                    y,
                    x=xi,
                    special_values=special_values,
                    print_info=False,
                    bin_num_limit=num)[xi]
                # check 单调性
                gb_drop_special_values = gb.loc[gb["is_special_values"] == False].reset_index(
                    drop=True)
                neg = (
                    (gb_drop_special_values['badprob'] -
                     gb_drop_special_values['badprob'].shift(1)).dropna() > badratediff).all()
                pos = ((gb_drop_special_values['badprob'] -
                       gb_drop_special_values['badprob'].shift(-1)).dropna() > badratediff).all()
                gb_num = gb_drop_special_values.shape[0]
                if neg or pos or gb_num <= 2:
                    bins[xi] = gb
                    break
        except BaseException as e:
            print(f"{xi} error!", e)
    return bins


# 获取切割点
def get_split_dict(bins):
    """
    获取分箱切割点
    :param bins: dict，分享结果
    :return:
    """
    split_dict = {}
    for var, tmp in bins.items():
        split_points = bins[var].loc[~bins[var]
                                     ["is_special_values"], "breaks"].tolist()
        if len(split_points) == 0:
            split_dict[var] = [np.inf]
        else:
            split_dict[var] = split_points
    return split_dict


# 训练集DEV的IV和PSI
def get_iv_psi_data(dev, oot, y, x=None,
                    bins=None,
                    special_values=None,
                    badratediff=0,
                    iv_threshold=0.02,
                    psi_threshold=0.15):
    if bins is None:
        bins = mono_woebin(
            dev,
            y,
            x,
            special_values=special_values,
            badratediff=badratediff)
    split_dict = get_split_dict(bins)
    oot_bins = sc.woebin(
        oot,
        y,
        x,
        special_values=special_values,
        breaks_list=split_dict,
        print_info=False)

    train_bins_data = pd.concat(
        bins,
        ignore_index=True).add_prefix(
        "train_",
    )
    oot_bins_data = (
        pd.concat(oot_bins, ignore_index=True)
        .drop(["breaks", "is_special_values"], axis=1)
        .add_prefix("oot_")
    )
    bins_data = (
        train_bins_data.merge(
            oot_bins_data,
            left_on=["train_variable", "train_bin"],
            how="outer",
            right_on=["oot_variable", "oot_bin"],
        )
    ).rename(columns={"train_variable": "variable", "train_bin": "bin"})

    # 计算psi
    bins_data = bins_data.assign(
        train_oot_psi=lambda k: (k.train_count_distr - k.oot_count_distr)
        * np.log((k.train_count_distr + 0.001) / (k.oot_count_distr + 0.001))
    )
    bins_data = bins_data.drop(
        ["oot_variable", "oot_bin"], axis=1
    )

    iv_psi_data = (
        bins_data.groupby(by="variable")
        .apply(
            lambda k: pd.Series(
                {
                    "train_total_iv": k.train_total_iv.tolist()[0],
                    "oot_total_iv": k.oot_total_iv.tolist()[0],
                    # "train_max_ks": k.train_max_ks.tolist()[0],
                    # "oot_max_ks": k.oot_max_ks.tolist()[0],
                    "train_oot_psi": k.train_oot_psi.sum(),
                }
            )
        )
        .sort_values(
            by=["train_total_iv", "train_oot_psi"],
            ascending=[False, False],
        )
        .reset_index()
    )

    del_features_by_ivpsi = iv_psi_data.loc[
        (iv_psi_data["train_total_iv"] < iv_threshold)
        # | (iv_psi_data["train_test_psi"] > psi_threshold)
        | (iv_psi_data["train_oot_psi"] > psi_threshold),
        "variable",
    ].tolist()
    return {
        "bins": bins,
        "bins_data": bins_data,
        "iv_psi_data": iv_psi_data,
        "del_features_by_ivpsi": del_features_by_ivpsi,
    }


def select_feature_by_corr_woe(
    iv_psi_data, iv_str, train_woe, corr_thresh=0.7,
):
    iv_psi_data = iv_psi_data.sort_values(
        by=[iv_str],
        ascending=False).reset_index(
        drop=True)
    need_features = iv_psi_data["variable"].tolist()
    del_features_by_corr = []
    for i in range(len(need_features)):
        x1 = need_features[i] + "_woe"
        for j in range(i + 1, len(need_features)):
            x2 = need_features[j] + "_woe"
            if x1 not in del_features_by_corr:
                corr = train_woe[[x1, x2]].corr().iloc[0, 1]
                if np.abs(corr) > corr_thresh:
                    del_features_by_corr.append(x2)
    return list(set(del_features_by_corr))


def select_feature_by_corr(
    iv_psi_data, iv_str, train, y, corr_thresh=0.7, is_plot=True
):
    iv_psi_data = iv_psi_data.sort_values(
        by=[iv_str],
        ascending=False).reset_index(
        drop=True)
    need_features = iv_psi_data["variable"].tolist()
    del_features_by_corr = []
    for i in range(len(need_features)):
        x1 = need_features[i]
        for j in range(i + 1, len(need_features)):
            x2 = need_features[j]
            if x1 not in del_features_by_corr:
                corr = train[[x1, x2]].corr().iloc[0, 1]
                if np.abs(corr) > corr_thresh:
                    del_features_by_corr.append(x2)
    return list(set(del_features_by_corr))


def select_features_by_feature_importance(
        data, y, importance_type="total_gain"):
    """
    xgboost特征的重要性
    :param data: frame
    :param y: string
    :param importance_type: string，特征重要性类型
    :return:
    """
    model = XGBClassifier(
        learning_rate=0.1,
        min_child_weight=20,
        max_depth=3,
        n_estimators=100,
        random_state=1,
        verbosity=0
    )
    X_train = data.drop(y, axis=1)
    y_train = data[y]
    model.fit(X_train, y_train)
    fea_imp = pd.DataFrame({"var": X_train.columns})
    model.importance_type = importance_type
    fea_imp[importance_type] = model.feature_importances_
    return fea_imp.sort_values(
        by=importance_type,
        ascending=False).reset_index(
        drop=True)


def get_vif(data, y):
    X = data.drop(y, axis=1)
    vif_data = (
        pd.DataFrame(
            {"feature": X.columns, "vif": [
                VIF(X.values, i) for i in range(X.shape[1])]}
        )
        .sort_values(by="vif", ascending=False)
        .reset_index(drop=True)
    )

    return vif_data


def select_feature_by_vif(data, y, threshold=5):
    vif_data = get_vif(data=data, y=y)
    need_features = data.drop(y, axis=1).columns.tolist()
    vif_del_res = {}
    while vif_data.vif.max() > threshold and len(need_features) > 2:
        del_feature = vif_data.loc[vif_data.vif.idxmax(), "feature"]
        vif_del_res[del_feature] = vif_data.vif.max()
        need_features.remove(del_feature)
        vif_data = get_vif(data[need_features + [y]], y=y)
    return {"vif_data": vif_data, "vif_del_res": vif_del_res}


def calculate_iv_ks(data: DataFrame, bin: str, y: str,
                    is_score: bool = True) -> DataFrame:
    data = data.copy()
    default_bad_ratio = data[y].mean()
    gb = data.groupby(by=bin, as_index=False)[
        y].agg({"good": n0, "bad": n1})
    if not is_score:
        gb = gb.sort_index(ascending=False)
    gb = (
        gb.assign(
            group=lambda x: x.good +
            x.bad).assign(
            group_ratio=lambda x: x.group /
            x.group.sum(),
            bad_ratio=lambda x: x.bad /
            x.group,
            cum_bad_rate=lambda x: x.bad.cumsum() /
            x.bad.sum(),
            cum_good_rate=lambda x: x.good.cumsum() /
            x.good.sum(),
        ).assign(
            lift=lambda x: x.bad_ratio /
            default_bad_ratio,
            cum_lift=lambda x: (
                x.bad.cumsum() /
                x.group.cumsum()) /
            default_bad_ratio,
            ks=lambda x: np.abs(
                x.cum_bad_rate -
                x.cum_good_rate),
            iv=lambda x: (
                x.bad /
                x.bad.sum() -
                x.good /
                x.good.sum()) *
            np.log(
                ((x.bad + 0.001) /
                 x.bad.sum()) /
                (
                    (x.good + 0.001) /
                    x.good.sum())),
        ).assign(
            max_ks=lambda x: np.abs(
                x.ks).max(),
            total_iv=lambda x: x.iv.sum()))

    return gb


def plot_auc_ks(gb: DataFrame, auc: float) -> None:
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 110
    x1 = [0] + gb["cum_good_rate"].tolist()
    y1 = [0] + gb["cum_bad_rate"].tolist()
    plt.plot(x1, y1)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.text(0.85, 0.05, f"auc:{round(auc, 4)}")
    plt.fill_between(x1, y1, color="grey", alpha=0.3)
    plt.title("ROC曲线")
    plt.xlabel("累计好样本占比")
    plt.ylabel("累计坏样本占比")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.show()

    # ks
    x2 = [0] + gb["group_ratio"].cumsum().tolist()
    y21 = [0] + gb["cum_bad_rate"].tolist()
    y22 = [0] + gb["cum_good_rate"].tolist()
    y23 = [0] + (gb["cum_bad_rate"] - gb["cum_good_rate"]).tolist()
    plt.plot(x2, y21, label="fpr")
    plt.plot(x2, y22, label="tpr")
    plt.plot(x2, y23, label="ks")
    max_ks = gb.ks.max()
    max_ks_idx = gb.ks.idxmax()
    max_ks_group_ratio = gb.loc[0:max_ks_idx, "group_ratio"].sum()
    plt.text(
        max_ks_group_ratio + 0.02,
        max_ks + 0.02,
        f"ks:{round(max_ks, 4)}")
    plt.plot([max_ks_group_ratio, max_ks_group_ratio],
             [0, max_ks], "--", color="grey")
    plt.title("K-S曲线")
    plt.xlabel("累计样本占比")
    plt.ylabel("累计好/坏样本占比")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.01, 0.1))
    # plt.savefig("K-S曲线.png")
    plt.legend()
    plt.show()


def calculate_cut_off_data(
        data: DataFrame, y: str, score: str, split_points: list, is_score: bool = True, is_plot: bool = True) -> DataFrame:
    split_points = [float("-inf")] + split_points + [float("inf")]
    data = data.copy()
    data["bin"] = pd.cut(data[score], bins=split_points, right=False)
    gb1 = calculate_iv_ks(
        data=data,
        bin="bin",
        y=y,
        is_score=is_score)
    if is_plot:
        gb2 = calculate_iv_ks(
            data=data,
            bin=score,
            y=y,
            is_score=is_score)
        auc = roc_auc_score(data[y], -data[score])
        plot_auc_ks(gb2, auc)
    return gb1


def get_auc_ks(y_true, y_proba, is_print=True):
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = np.abs(tpr - fpr).max()
    if is_print:
        print("auc: \n", auc)
        print("ks: \n", ks)
    return {"auc": auc, "ks": ks}
