# !/usr/bin/env python
# coding=utf-8

# @Time    : 2022/10/9 14:16
# @Author  : pineapple
# @File    : scorecard2python
# @Software: PyCharm


import re
import pandas as pd


def scorecard_if_print(data, special_value=-999, special_score=0):
    data = data.copy().reset_index()
    variable = data.loc[0, "variable"]
    text = f"\t# {variable}\n"
    x = data.loc[0, "variable_no"]
    bin_list = data["bin"].tolist()
    for i in data.index:
        bin_min = data.loc[i, "bin_min"]
        bin_max = data.loc[i, "bin_max"]
        score = data.loc[i, "score"]
        if i == 0:
            text += f"\tif {bin_min} <= {x} < {bin_max}:\n\t\t{x}_score = {score}\n"
        elif 0 < i < data.shape[0]-1:
            text += f"\telif {bin_min} <= {x} < {bin_max}:\n\t\t{x}_score = {score}\n"
        else:
            text += f"\telif {x} >= {bin_min}:\n\t\t{x}_score = {score}\n"
        if bin_min == special_value:
            continue

    if special_value in bin_list:
        score = data.loc[data["bin_min"] == special_value, "score"].values[0]
        text += f"\telif {x} == {special_value}:\n\t\t{x}_score = {score}\n"
    else:
        text += f"\telif {x} == {special_value}:\n\t\t{x}_score = {special_score}\n"
    text += f"\telif {x} == -998:\n\t\t{x}_score = 0\n\telse:\n\t\t{x}_score = 0\n"
    text += f"\tvar_score_list.append({x}_score)\n\n"
    return text.replace("-inf", "0")


def scorecard2python(data, special_value="-999.0",
                     special_score=0, model_name="test"):
    data["bin_min"] = data["bin"].apply(
        lambda k: float(
            re.findall(
                "\\[(.+?),(.+?)\\)",
                k)[0][0]) if k != "" and k != str(special_value) and k != special_value and not pd.isnull(
            k) else special_value)
    data["bin_max"] = data["bin"].apply(
        lambda k: float(re.findall("\\[(.+?),(.+?)\\)", k)[0][1])
        if k != "" and k != str(special_value) and k != special_value and not pd.isnull(k) else special_value)

    text = f"def {model_name}(input_dict):\n"
    text += f'\t"""{special_value}为传入值，-998为超时"""\n'
    x_list = data["variable_no"].unique().tolist()

    # 截距
    x0_score = data.loc[data["variable_no"] == "x0", "score"].values[0]
    text += f"\t# 变量值列表\n\tvar_score_list = []\n\n\t# 截距\n\tx0_score = {x0_score}\n\n\t# 获取变量\n"
    text += f"\tvar_score_list.append(x0_score)\n\n"

    x_list.remove("x0")
    # 获取各个变量
    for x in x_list:
        text += f"\t{x} = eval(input_dict['{x}'])\n"

    for x in x_list:
        data_tmp = data.loc[data["variable_no"] == x]
        text += scorecard_if_print(data_tmp, special_value=special_value, special_score=special_score)

    text += "\tmodel_score = sum(var_score_list)\n\treturn var_score_list, model_score\n"
    text = text.replace("\t", "    ")
    import os
    file_name = model_name+".py"
    if file_name in os.listdir():
        os.remove(file_name)
    with open(file_name, "w") as f:
        f.write(text)
    f.close()

