import os
import re
import pandas as pd


def data_flag(data_out):
    data_output_flag = data_out[:][0].tolist()
    return data_output_flag


def data_seq(data_out):
    data_out = data_out[:][1]
    split = data_out.str.split(':', expand=True)
    return split


def findnumber(seq):
    out_seq = []
    p = re.compile(r'[(](.*?)[)]')  # find out the content in ()#
    for item in seq[1:]:
        if item is not None:
            temp = re.findall(p, item)
            temp1 = ''.join(temp)
            temp2 = temp1.split(',')
            temp2 = list(map(eval, temp2))
            out_seq.append(temp2)
    return out_seq


# 计算传递参数S
def cal_s(item):
    s = 0
    if len(item) == 5:
        s = item[4] * (item[3] + item[2] - item[1] - item[0])
    return s


# 检测每个C序列
def check_c(ind, inputs, out_pred, s_c, s_d):
    item = out_pred[ind]
    p = inputs
    flag = True

    s_c += cal_s(item)

    # 不等式判断条件
    if not (item[0] <= item[1] <= item[2] <= item[3]):
        flag = False
    if not (p[3] + 0.5 * s_c - p[0] * p[1] * s_d <= p[5]):
        flag = False
    if not (p[6] + item[4] <= p[7]):
        flag = False
    if ind > 1:
        item_pre = out_pred[ind - 1]
        if (item_pre[0] + item_pre[1] * (p[1] + p[2]) - p[2]) > item[0]:
            flag = False
    if not item[3] <= p[12]:
        flag = False
    if not item[4] <= p[10]:
        flag = False
    '''    
    if not (item[4] <= p[8] * (item[1] - item[0])):
        flag = False
    if not (item[4] <= p[9] * (item[3] - item[2])):
        flag = False
    '''
    return flag, s_c


# 检测每个D序列
def check_d(ind, inputs, out_pred, s_c, s_d):
    item = out_pred[ind]
    p = inputs
    flag = True

    s_d += item[1]

    if not (p[3] + 0.5 * s_c - p[0] * p[1] * s_d >= p[4]):
        flag = False
    if not (item[0] + item[1] * (p[1] + p[2]) - p[2] <= p[12]):
        flag = False
    if ind > 1:
        item_pre1 = out_pred[ind - 1]
        if item_pre1[3] > item[0]:
            flag = False
    if ind > 2:
        item_pre2 = out_pred[ind - 2]
        if (item[0] - (item_pre2[0] + item_pre2[1] * (p[1] + p[2]) - p[2])) < p[11]:
            flag = False

    return flag, s_d


# 检查每一条预测输出
def eval_single(inputs, feature, pred):
    num = len(pred)
    p = inputs
    s_c = 0
    s_d = 0
    total_flag = True

    for ind in range(num):
        if len(pred[ind]) == 5:
            flag, s_c = check_c(ind, p, pred, s_c, s_d)
            total_flag = (total_flag and flag)
        elif len(pred[ind]) == 2:
            flag, s_d = check_d(ind, p, pred, s_c, s_d)
            total_flag = (total_flag and flag)

    if not total_flag:
        return "Infeasible"
    elif total_flag and feature != s_d:
        return "Feasible"
    elif total_flag and feature == s_d:
        return "Optimal"


# 检查预测输出txt
def eval_all(inputs, feature, pred):
    if len(inputs) == len(feature) == len(pred):
        eval_flag = True
    else:
        eval_flag = False

    eval_result = []
    for ind in range(len(inputs)):
        eval_input = inputs[ind]
        feature_value = feature[ind]
        out_pred = findnumber(pred[ind])
        eval_result.append(eval_single(eval_input, feature_value, out_pred))

    if eval_flag:
        return eval_result
    elif not eval_flag:
        return None


def evaluate(input_file, output_file, true_file, result_file):

    data_input = pd.read_csv('./'+input_file, sep=',', header=None)      # 输入文件
    data_true = pd.read_csv('./'+true_file, sep='|', header=None)       # PHD软件生成输出
    data_pred = pd.read_csv('./'+output_file, sep='|', header=None)       # 模型预测输出

    data_input = data_input.values.tolist()
    feature_true = data_flag(data_true)
    seq_pred = data_seq(data_pred)
    seq_pred = seq_pred.values.tolist()

    result = eval_all(data_input, feature_true, seq_pred)

    fw = open('./' + result_file, 'w')
    for item in result:
        fw.write(item + "\n")

    optimal = 0
    feasible = 0
    infeasible = 0

    for i in result:
        if i == "Optimal":
            optimal += 1
        elif i == "Feasible":
            feasible += 1
        elif i == "Infeasible":
            infeasible += 1
    print("Optimal:", optimal, "|", "Feasible:", feasible, "|", "Infeasible:", infeasible)

    # Availability of models
    Availability = optimal + feasible
    print("Availability of models: %.4f" % (Availability / len(result)))
