from scorecard2python import scorecard2python
import pandas as pd


if __name__ == '__main__':
    special_value = "-999.0"
    data = pd.read_excel("/Users/luoyifan/内部工作/2023-08/小模型/modelReport/小模型最终版20230809.xlsx", sheet_name="评分卡")
    data = data.drop("Unnamed: 0", axis=1)
    print("scorecard:\n", data)
    scorecard2python(data, model_name="sh_bigdata_small_model1_v1")
    from sh_bigdata_small_model1_v1 import sh_bigdata_small_model1_v1
    input_dict = {
        "x1": "700", "x2": "10", "x3": "400", "x4": "20"
    }
    print("input dict: \n", input_dict)
    print("评分结果: \n", sh_bigdata_small_model1_v1(input_dict))
