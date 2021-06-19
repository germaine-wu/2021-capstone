import os
import sys

import joblib
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from config import loadData, find

warnings.filterwarnings("ignore")

'''
This section is to train models of method 2 from Sample Data sets.
There are eight models to form the method 2 model structure.
1.Feature model
2.Start C/D model
3.C1 model
4.C4 model
5.Model for C2,C3(C2 == C3)
6.C5 model
7.D1 model
8.D2 model

Pre-trained models already exist.
No need to run model2.py if predict output.
Please backup the existed method 2 models before running model2.py! 
'''

# Create Dir to same model1 (if not exist)
path = './model/model2'
if not os.path.exists(path):
    os.makedirs(path)
sys.stdout.write("0%\r")
sys.stdout.flush()

############ Load Training Data set ############

data_sample, data_output = loadData()
data_sample = data_sample.values.tolist()
sys.stdout.write("10%\r")
sys.stdout.flush()

############ Preprocess of Sample Data set ############


def data_flag(data_out):
    flag = data_out[:][0].tolist()
    return flag


def data_seq(data_out):
    data_out = data_out[:][1]
    split = data_out.str.split(':', expand=True)
    return split


def divided_seq(sample, lst, flag):
    # Divide Train set and Output set
    seq1 = find(lst[1].tolist())    # 1st sequence of sample output
    seq2 = find(lst[2].tolist())    # 2nd sequence of sample output
    seq3 = find(lst[3].tolist())    # 3rd sequence of sample output
    seq4 = find(lst[4].tolist())    # 4th sequence of sample output

    data_output_seqtype = lst[0]
    data_output_seqtype = data_output_seqtype.tolist()
    # Type: ONE C item
    input_one = []
    output_one = []
    # Type C sequence
    input_c = []
    flag_c = []
    output_c1 = []
    output_c2 = []
    output_c3 = []
    output_c4 = []
    # Type D sequence
    input_d = []
    output_d1 = []
    output_d2 = []
    output_d3 = []
    output_d4 = []
    # Valid
    split_buffer = []

    for i in range(0, len(sample)):
        if flag[i] == 0:
            input_one.append(sample[i])
            output_one.append(seq1[i])
        elif flag[i] != 0:
            if data_output_seqtype[i] == "C":
                input_c.append(sample[i])
                flag_c.append(flag[i])
                output_c1.append(seq1[i])
                output_c2.append(seq2[i])
                output_c3.append(seq3[i])
                output_c4.append(seq4[i])
            elif data_output_seqtype[i] == "D":
                input_d.append(sample[i])
                output_d1.append(seq1[i])
                output_d2.append(seq2[i])
                output_d3.append(seq3[i])
                output_d4.append(seq4[i])
            else:
                split_buffer.append(seq1[i])
        else:
            split_buffer.append(seq1[i])

    if not split_buffer:
        sys.stdout.write("20%\r")
        sys.stdout.flush()
        return input_one, output_one, input_c, flag_c, output_c1, output_c2, output_c3, output_c4, input_d, output_d1, output_d2, output_d3, output_d4
    else:
        print("Error - Preprocess Training Data!")
        return None


data_output_flag = data_flag(data_output)
data_output_split = data_seq(data_output)
input_one,output_one,input_c,flag_c,output_c1,output_c2,output_c3,output_c4,input_d,output_d1,output_d2,output_d3,output_d4 = divided_seq(data_sample,data_output_split,data_output_flag)

# sequence divided
def C_divided(C_seq):
    c1, c2, c3, c4, c5 = [], [], [], [], []
    for i in range(len(C_seq)):
        c1.append(C_seq[i][0])
        c2.append(C_seq[i][1])
        c3.append(C_seq[i][2])
        c4.append(C_seq[i][3])
        c5.append(C_seq[i][4])
    return c1, c2, c3, c4, c5


def D_divided(D_seq):
    d1, d2 = [], []
    for i in range(len(D_seq)):
        d1.append(D_seq[i][0])
        d2.append(D_seq[i][1])
    return d1, d2


# input reduce
def input_red(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[1]+i[2]]+[i[2]]+[i[11]]+[i[12]]+[i[3]-i[4]]+[i[5]-i[4]]+[i[7]-i[6]]+[i[8]]+[i[9]]+[i[10]])
    return new_i_input


def input_red_c1(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[1]+i[2]]+[i[2]]+[i[3]-i[4]]+[i[5]-i[4]])
    return new_i_input


def input_red_c23(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[3]-i[4]]+[i[5]-i[4]]+[i[7]-i[6]]+[i[8]]+[i[9]]+[i[10]])
    return new_i_input


def input_red_c4(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[1]+i[2]]+[i[2]]+[i[3]-i[4]]+[i[5]-i[4]]+[i[12]])
    return new_i_input


def input_red_c5(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[7]-i[6]]+[i[8]]+[i[9]]+[i[10]])
    return new_i_input


def input_red_d2(inputs):
    new_i_input = []
    for i in inputs:
        new_i_input.append([i[0]*i[1]]+[i[1]+i[2]]+[i[2]]+[i[11]]+[i[12]])
    return new_i_input


# Calculation of  S_C & S_D
def S_C_cal(output_c):
    return output_c[4]*(output_c[3]+output_c[2]-output_c[1]-output_c[0])


def S_D_cal(output_d):
    return output_d[1]


def data_pre_C_seq(input_c, output_c1, output_c2, output_c3, output_c4):
    input_c_new_c1 = input_red_c1(input_c)
    input_c_new_c23 = input_red_c23(input_c)
    input_c_new_c4 = input_red_c4(input_c)
    input_c_new_c5 = input_red_c5(input_c)
    input_c_new_d2 = input_red_d2(input_c)
    input_c1_train = []
    input_c4_train = []
    input_c23_train = []
    input_c5_train = []
    input_d1_train = []
    input_d2_train = []
    output_c1_train = []
    output_c4_train = []
    output_c23_train = []
    output_c5_train = []
    output_d1_train = []
    output_d2_train = []
    for i in range(len(input_c)):
        input_c1_train.append(input_c_new_c1[i]+[0]+[-1]+[-1])
        input_c4_train.append(input_c_new_c4[i]+[0]+[-1]+[-1]+[output_c1[i][0]])
        input_c23_train.append(input_c_new_c23[i]+[0]+[output_c1[i][0]]+[output_c1[i][3]])
        input_c5_train.append(input_c_new_c5[i]+[0]+[output_c1[i][1]-output_c1[i][0]]+[output_c1[i][3]-output_c1[i][2]])
        output_c1_train.append(output_c1[i][0])
        output_c4_train.append(output_c1[i][3])
        output_c23_train.append(output_c1[i][1])
        output_c5_train.append(output_c1[i][4]*10000)
        if output_c2[i] != None:
            S = 0.5*S_C_cal(output_c1[i])
            input_d1_train.append(input_c_new_d2[i]+[S]+[-1]+[-1])
            input_d2_train.append(input_c_new_d2[i]+[S]+[-1]+[-1]+[output_c2[i][0]])
            output_d1_train.append(output_c2[i][0])
            output_d2_train.append(output_c2[i][1])
        if output_c3[i] != None:
            S = 0
            S = S + 0.5*S_C_cal(output_c1[i])
            S = input_c_new_c1[i][0] * S_D_cal(output_c2[i]) - S
            input_c1_train.append(input_c_new_c1[i]+[S]+[output_c2[i][0]]+[output_c2[i][1]])
            input_c4_train.append(input_c_new_c4[i]+[S]+[output_c2[i][0]]+[output_c2[i][1]]+[output_c3[i][0]])
            input_c23_train.append(input_c_new_c23[i]+[S]+[output_c3[i][0]]+[output_c3[i][3]])
            input_c5_train.append(input_c_new_c5[i]+[S]+[output_c3[i][1]-output_c3[i][0]]+[output_c3[i][3]-output_c3[i][2]])
            output_c1_train.append(output_c3[i][0])
            output_c4_train.append(output_c3[i][3])
            output_c23_train.append(output_c3[i][1])
            output_c5_train.append(output_c3[i][4]*10000)
        if output_c4[i] != None:
            S = input_c_new_d2[i][0] * S_D_cal(output_c2[i])
            S = 0.5*(S_C_cal(output_c1[i]) + S_C_cal(output_c3[i])) - S
            input_d1_train.append(input_c_new_d2[i]+[S]+[output_c2[i][0]]+[output_c2[i][1]])
            input_d2_train.append(input_c_new_d2[i]+[S]+[output_c2[i][0]]+[output_c2[i][1]]+[output_c4[i][0]])
            output_d1_train.append(output_c4[i][0])
            output_d2_train.append(output_c4[i][1])
    sys.stdout.write("30%\r")
    sys.stdout.flush()
    return input_c1_train,output_c1_train,input_c4_train,output_c4_train,input_c23_train,output_c23_train,input_c5_train,output_c5_train,input_d1_train,output_d1_train,input_d2_train,output_d2_train


def data_pre_D_seq(input_d, output_d1, output_d2, output_d3, output_d4):
    #input_d_new = input_red_d2(input_d)
    input_d_new_c1 = input_red_c1(input_d)
    input_d_new_c23 = input_red_c23(input_d)
    input_d_new_c4 = input_red_c4(input_d)
    input_d_new_c5 = input_red_c5(input_d)
    input_d_new_d2 = input_red_d2(input_d)
    input_c1_train_d = []
    input_c4_train_d = []
    input_c23_train_d = []
    input_c5_train_d = []
    input_d1_train_d = []
    input_d2_train_d = []
    output_c1_train_d = []
    output_c4_train_d = []
    output_c23_train_d = []
    output_c5_train_d = []
    output_d1_train_d = []
    output_d2_train_d = []
    for i in range(len(input_d)):
        S = 0
        input_d1_train_d.append(input_d_new_d2[i]+[S]+[-1]+[-1])
        input_d2_train_d.append(input_d_new_d2[i]+[S]+[-1]+[-1]+[output_d1[i][0]])
        output_d1_train_d.append(output_d1[i][0])
        output_d2_train_d.append(output_d1[i][1])
        if output_d2[i] != None:
            S = input_d_new_c1[i][0] * S_D_cal(output_d1[i])
            input_c1_train_d.append(input_d_new_c1[i]+[S]+[output_d1[i][0]]+[output_d1[i][1]])
            input_c4_train_d.append(input_d_new_c4[i]+[S]+[output_d1[i][0]]+[output_d1[i][1]]+[output_d2[i][0]])
            input_c23_train_d.append(input_d_new_c23[i]+[S]+[output_d2[i][0]]+[output_d2[i][3]])
            input_c5_train_d.append(input_d_new_c5[i]+[S]+[output_d2[i][1]-output_d2[i][0]]+[output_d2[i][3]-output_d2[i][2]])
            output_c1_train_d.append(output_d2[i][0])
            output_c4_train_d.append(output_d2[i][3])
            output_c23_train_d.append(output_d2[i][1])
            output_c5_train_d.append(output_d2[i][4]*10000)
        if output_d3[i] != None:
            S = 0.5*S_C_cal(output_d2[i])-input_d_new_c1[i][0] * S_D_cal(output_d1[i])
            input_d1_train_d.append(input_d_new_d2[i]+[S]+[output_d1[i][0]]+[output_d1[i][1]])
            input_d2_train_d.append(input_d_new_d2[i]+[S]+[output_d1[i][0]]+[output_d1[i][1]]+[output_d3[i][0]])
            output_d1_train_d.append(output_d3[i][0])
            output_d2_train_d.append(output_d3[i][1])
        if output_d4[i] != None:
            S = input_d_new_c1[i][0]*(S_D_cal(output_d1[i])+S_D_cal(output_d3[i])) - 0.5*S_C_cal(output_d2[i])
            input_c1_train_d.append(input_d_new_c1[i]+[S]+[output_d3[i][0]]+[output_d3[i][1]])
            input_c4_train_d.append(input_d_new_c4[i]+[S]+[output_d3[i][0]]+[output_d3[i][1]]+[output_d4[i][0]])
            input_c23_train_d.append(input_d_new_c23[i]+[S]+[output_d4[i][0]]+[output_d4[i][3]])
            input_c5_train_d.append(input_d_new_c5[i]+[S]+[output_d4[i][1]-output_d4[i][0]]+[output_d4[i][3]-output_d4[i][2]])
            output_c1_train_d.append(output_d4[i][0])
            output_c4_train_d.append(output_d4[i][3])
            output_c23_train_d.append(output_d4[i][1])
            output_c5_train_d.append(output_d4[i][4]*10000)
    sys.stdout.write("40%\r")
    sys.stdout.flush()
    return input_c1_train_d, output_c1_train_d, input_c4_train_d, output_c4_train_d,input_c23_train_d,output_c23_train_d,input_c5_train_d,output_c5_train_d,input_d1_train_d,output_d1_train_d,input_d2_train_d,output_d2_train_d


input_c1_train,output_c1_train,input_c4_train,output_c4_train,input_c23_train,output_c23_train,input_c5_train,output_c5_train,input_d1_train,output_d1_train,input_d2_train,output_d2_train= data_pre_C_seq(input_c,output_c1,output_c2,output_c3,output_c4)
input_c1_train_d,output_c1_train_d,input_c4_train_d,output_c4_train_d,input_c23_train_d,output_c23_train_d,input_c5_train_d,output_c5_train_d,input_d1_train_d,output_d1_train_d,input_d2_train_d,output_d2_train_d = data_pre_D_seq(input_d,output_d1,output_d2,output_d3,output_d4)

# Reduce input for Feature & seq_type models
new_i_train = []
for i in data_sample:
    new_i_train.append([i[0]*i[1]]+[i[1]+i[2]]+[i[2]+i[12]]+[i[3]-i[4]]+[i[5]-i[4]])


############ Prediction models ############

# 1.Feature model
model_feature = DecisionTreeRegressor(random_state=0)
model_feature.fit(new_i_train, data_output_flag)
sys.stdout.write("50%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/feature.model', value=model_feature)

# 2.Start C/D model
start_CD = data_output_split[0]
model_CD = DecisionTreeClassifier(random_state=0)
model_CD.fit(new_i_train, start_CD)
sys.stdout.write("55%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/CD.model', value=model_CD)

# 3.C1 model
input_c1_final = input_c1_train + input_c1_train_d
output_c1_final = output_c1_train + output_c1_train_d

model_C1 = DecisionTreeRegressor(random_state=0)
model_C1.fit(input_c1_final, output_c1_final)
sys.stdout.write("60%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/C1.model', value=model_C1)

# 4.C4 model
input_c4_final = input_c4_train + input_c4_train_d
output_c4_final = output_c4_train + output_c4_train_d

model_C4 = DecisionTreeRegressor(random_state=0)
model_C4.fit(input_c4_final, output_c4_final)
sys.stdout.write("70%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/C4.model', value=model_C4)

# 5.Model for C2,C3(C2 == C3)
input_c23_final = input_c23_train + input_c23_train_d
output_c23_final = output_c23_train + output_c23_train_d

model_C23 = DecisionTreeRegressor(random_state=0)
model_C23.fit(input_c23_final, output_c23_final)
sys.stdout.write("80%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/C23.model', value=model_C23)

# 6.C5 model
input_c5_final = input_c5_train + input_c5_train_d
output_c5_final = output_c5_train + output_c5_train_d

model_C5 = DecisionTreeRegressor(random_state=0)
model_C5.fit(input_c5_final, output_c5_final)
sys.stdout.write("90%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/C5.model', value=model_C5)

'''
# 7.D1 model
input_d1_final = output_c4_train + output_c4_train_d
output_d1_final = output_d1_train + output_d1_train_d

model_D1 = DecisionTreeRegressor(random_state=0)
model_D1.fit(input_d1_final, output_d1_final)
joblib.dump(filename='/models/D1.model', value=model_D1)
'''

# 8.D2 model
input_d2_final = input_d2_train + input_d2_train_d
output_d2_final = output_d2_train + output_d2_train_d

model_D2 = DecisionTreeRegressor(random_state=0)
model_D2.fit(input_d2_final, output_d2_final)
sys.stdout.write("100%\r")
sys.stdout.flush()
joblib.dump(filename=path+'/D2.model', value=model_D2)

