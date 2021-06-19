import os
import pandas as pd
import joblib

# Load pre-trained models
global model1_flag
global model2_flag

if not os.path.exists('./model/model1'):
    print("Run model1.py to train models first!")
    model1_flag = False
else:
    path_model1 = './model/model1'
    model_number = joblib.load(filename=path_model1 + '/number.model')
    model_1_seqtype = joblib.load(filename=path_model1 + '/model_1_seqtype.model')
    model_1c_seqdata = joblib.load(filename=path_model1 + '/model_1c_seqdata.model')
    model_1d_seqdata = joblib.load(filename=path_model1 + '/model_1d_seqdata.model')
    model_2_seqtype = joblib.load(filename=path_model1 + '/model_2_seqtype.model')
    model_2dseq = joblib.load(filename=path_model1 + '/model_2dseq.model')
    model_2cseq = joblib.load(filename=path_model1 + '/model_2cseq.model')
    model_3_seqtype = joblib.load(filename=path_model1 + '/model_3_seqtype.model')
    model_3d_seqdata = joblib.load(filename=path_model1 + '/model_3d_seqdata.model')
    model1_flag = True

if not os.path.exists('./model/model2'):
    print("Run model2.py to train models first!")
    model2_flag = False
else:
    path_model2 = './model/model2'
    model_feature = joblib.load(filename=path_model2 + '/feature.model')
    model_CD = joblib.load(filename=path_model2 + '/CD.model')
    model_C1 = joblib.load(filename=path_model2 + '/C1.model')
    model_C4 = joblib.load(filename=path_model2 + '/C4.model')
    model_C23 = joblib.load(filename=path_model2 + '/C23.model')
    model_C5 = joblib.load(filename=path_model2 + '/C5.model')
    model_D2 = joblib.load(filename=path_model2 + '/D2.model')
    model2_flag = True


def predict_model1(inputs):
    pred = []

    if not model1_flag:
        return "Model1 not exist!"

    output_number = model_number.predict(inputs)
    inputs['13'] = output_number
    input_new = inputs.values.tolist()
    for i in range(len(input_new)):
        tem = []
        model_1_seqtype_pre = model_1_seqtype.predict([input_new[i]])

        if output_number[i] == 0:
            model_1c_seqdata_pre = model_1c_seqdata.predict([input_new[i]])
            tem.append("%d|%c:(%d,%f,%f,%d,%f)" % (
                output_number[i], model_1_seqtype_pre[0], list(model_1c_seqdata_pre)[0][0],
                list(model_1c_seqdata_pre)[0][1], list(model_1c_seqdata_pre)[0][2], list(model_1c_seqdata_pre)[0][3],
                list(model_1c_seqdata_pre)[0][4]))
            tem = ''.join(tem)
            pred.append(tem)

        elif model_1_seqtype_pre == "C":
            model_1c_seqdata_pre = model_1c_seqdata.predict([input_new[i]])
            input_new_1 = input_new[i] + [model_1c_seqdata_pre[0][0]] + [model_1c_seqdata_pre[0][1]] + [
                model_1c_seqdata_pre[0][2]] + [model_1c_seqdata_pre[0][3]] + [model_1c_seqdata_pre[0][4]]
            model_2dseq_pre = model_2dseq.predict([input_new_1])
            tem.append("%d|%c:(%d,%f,%f,%d,%f),%c:(%d,%d)" % (
                output_number[i], model_1_seqtype_pre[0], list(model_1c_seqdata_pre)[0][0],
                list(model_1c_seqdata_pre)[0][1], list(model_1c_seqdata_pre)[0][2], list(model_1c_seqdata_pre)[0][3],
                list(model_1c_seqdata_pre)[0][4], 'D', list(model_2dseq_pre)[0][0], list(model_2dseq_pre)[0][1]))
            tem = ''.join(tem)
            pred.append(tem)

        else:
            model_1d_seqdata_pre = model_1d_seqdata.predict([input_new[i]])
            if output_number[i] == model_1d_seqdata_pre[0][1]:
                tem.append("%d|%c:(%d,%d)" % (
                    output_number[i], model_1_seqtype_pre[0], list(model_1d_seqdata_pre)[0][0],
                    list(model_1d_seqdata_pre)[0][1]))
                tem = ''.join(tem)
                pred.append(tem)

            else:
                input_new_1 = input_new[i] + [model_1d_seqdata_pre[0][0]] + [model_1d_seqdata_pre[0][1]]

                model_2cseq_pre = model_2cseq.predict([input_new_1])
                input_new_12 = input_new_1 + [model_2cseq_pre[0][0]] + [model_2cseq_pre[0][1]] + [
                    model_2cseq_pre[0][2]] + [model_2cseq_pre[0][3]] + [model_2cseq_pre[0][4]]
                model_3d_seqdata_pre = model_3d_seqdata.predict([input_new_12])
                tem.append("%d|%c:(%d,%d),%c:(%d,%f,%f,%d,%f),%c:(%d,%d)" % (
                    output_number[i], model_1_seqtype_pre[0], list(model_1d_seqdata_pre)[0][0],
                    list(model_1d_seqdata_pre)[0][1], 'C', list(model_2cseq_pre)[0][0], list(model_2cseq_pre)[0][1],
                    list(model_2cseq_pre)[0][2], list(model_2cseq_pre)[0][3], list(model_2cseq_pre)[0][4], 'D',
                    list(model_3d_seqdata_pre)[0][0], list(model_3d_seqdata_pre)[0][1]))
                tem = ''.join(tem)
                pred.append(tem)

    print("Done!")
    return pred


def predict_model2(inputs):
    pred = []

    if not model2_flag:
        return "Model2 not exist!"

    for ind in range(len(inputs)):
        i = inputs[ind]
        temp = []
        input_f1 = input_divide_t(i)
        feature = model_feature.predict([input_f1])
        seq = model_CD.predict([input_f1])

        lst_c = [0, 0, 0, 0, 0]  # record last C seq
        lst_d = [-1, -1]  # record last C seq
        S_c = 0
        S_d = 0
        c_count = 0

        # 判断时候加upper bound
        res = int(feature[0])
        res = upper_bound(res, i)

        # check if the feature is 0, if yes, transfer to another model
        # C/D should be checked there
        if seq[0] == 'C':
            temp.append("%d|" % res)
            if res == 0:
                lst_c, tup_c, S_c = C_seq(i, lst_c, lst_d, S_c, res, c_count)
                temp.append(tup_c)
            while res > 0:
                lst_c, tup_c, S_c = C_seq(i, lst_c, lst_d, S_c, res, c_count)
                lst_d, tup_d, S_c = D_seq(i, lst_c, lst_d, S_c)

                res -= lst_d[1]
                temp.append(tup_c)
                if res >= 0:
                    temp.append(tup_d)
                elif res < 0:
                    tup_d = "D:(%d,%d)," % (lst_d[0], res + lst_d[1])
                    temp.append(tup_d)

            tem = ''.join(temp)
            tem = tem[:-1]
            pred.append(tem)

        elif seq[0] == 'D':
            temp.append("%d|" % res)
            lst_d, tup_d, S_d = D_seq(i, lst_c, lst_d, S_d)
            # 当输出只有1个D时，判断是否feature==D2
            res -= lst_d[1]
            if res >= 0:
                temp.append(tup_d)
            elif res < 0:
                tup_d = "D:(%d,%d)," % (lst_d[0], res + lst_d[1])
                temp.append(tup_d)

            while res > 0:
                lst_c, tup_c, S_d = C_seq(i, lst_c, lst_d, S_d, res, c_count)
                lst_d, tup_d, S_d = D_seq(i, lst_c, lst_d, S_d)

                res -= lst_d[1]
                temp.append(tup_c)
                if res >= 0:
                    temp.append(tup_d)
                elif res < 0:
                    tup_d = "D:(%d,%d)," % (lst_d[0], res + lst_d[1])
                    temp.append(tup_d)

            tem = ''.join(temp)
            tem = tem[:-1]
            pred.append(tem)

    print("Done!")
    return pred


def S_C_cal(output_c):
    return output_c[4] * (output_c[3] + output_c[2] - output_c[1] - output_c[0])


def S_D_cal(output_d):
    return output_d[1]


def input_divide_t(ind):
    return [ind[0] * ind[1]] + [ind[1] + ind[2]] + [ind[2] + ind[12]] + [ind[3] - ind[4]] + [ind[5] - ind[4]]


def C_seq(i, lst_c, lst_d, S, res, c_count):
    C1 = model_C1.predict(
        [[i[0] * i[1]] + [i[1] + i[2]] + [i[2]] + [i[3] - i[4]] + [i[5] - i[4]] + [S] + [lst_d[0]] + [lst_d[1]]])
    if C1[0] < lst_d[0] + lst_d[1] * (i[1] + i[2]) - i[2]:
        C1[0] = lst_d[0] + lst_d[1] * (i[1] + i[2]) - i[2]
    if lst_d[0] != -1 and C1[0] >= lst_d[0] + lst_d[1] * (i[1] + i[2]) - i[2]:
        C1[0] = lst_d[0] + lst_d[1] * (i[1] + i[2]) - i[2]

    C4 = model_C4.predict(
        [[i[0]] + [i[1]] + [i[2]] + [i[3] - i[4]] + [i[5] - i[4]] + [i[12]] + [S] + [lst_d[0]] + [lst_d[1]] + [C1]])
    if C4[0] >= i[12] - res * (i[1] + i[2]) + i[2] and c_count == 0:
        C4[0] = i[12] - res * (i[1] + i[2]) + i[2]
    elif c_count > 0:
        C4[0] = C1[0] + i[11]

    C23 = model_C23.predict([[i[0] * i[1]] + [i[3] - i[4]] + [i[5] - i[4]] + [i[7] - i[6]] + [i[8]] + [i[9]]
                             + [i[10]] + [S] + [C1] + [C4]])  # input_extent = input+C4
    C5 = model_C5.predict([[i[0] * i[1]] + [i[7] - i[6]] + [i[8]] + [i[9]] + [i[10]] + [S] + [C23 - C1] + [C4 - C23]])
    C5 = C5 / 10000

    item_c = [int(C1[0]), float('%.6f' % C23[0]), float('%.6f' % C23[0]), int(C4[0]), float('%.6f' % C5[0])]
    tup_c = "C:(%d,%.6f,%.6f,%d,%.6f)," % (C1[0], C23[0], C23[0], C4[0], C5[0])
    Sn = 0.5 * S_C_cal(item_c) - S
    c_count += 1
    return item_c, tup_c, Sn


def D_seq(i, lst_c, lst_d, S):
    D1 = lst_c[3]
    D2 = model_D2.predict(
        [[i[0] * i[1]] + [i[1] + i[2]] + [i[2]] + [i[11]] + [i[12]] + [S] + [lst_d[0]] + [lst_d[1]] + [D1]])

    item_d = [int(D1), int(D2[0])]
    tup_d = "D:(%d,%d)," % (D1, D2[0])
    Sn = i[0] * i[1] * S_D_cal(item_d) - S
    return item_d, tup_d, Sn


def upper_bound(res, inputs):
    while inputs[12] - res * (inputs[1] + inputs[2]) + inputs[2] <= 0 or res > (inputs[3] - inputs[4]) / (
            inputs[0] * inputs[1]):
        res -= 1
    return res


def prediction(input_file, output_file, method):
    test_x = pd.read_csv('./' + input_file, sep=',', header=None)

    if method == 1:
        pred = predict_model1(test_x)
    elif method == 2:
        test_x = test_x.values.tolist()
        pred = predict_model2(test_x)
    else:
        pred = 'Error prediction'

    fw = open('./' + output_file, 'w')
    for item in pred:
        fw.write(item + "\n")
