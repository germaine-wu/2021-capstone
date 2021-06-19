import os
import sys

import pandas as pd
import joblib
import re
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from config import loadData, find

warnings.filterwarnings("ignore")

'''
This section is to train models of method 1 from Sample Data sets.
There are nine models to form the method 1 model structure.
1.Feature model
2.Type model of 1st sequence
3.C sequence model of 1st sequence
4.D sequence model of 1st sequence
5.Type model of 2nd sequence
6.C sequence model of 2nd sequence
7.D sequence model of 2nd sequence
8.Type model of 3rd sequence
9.D sequence model of 3rd sequence

Pre-trained models already exist.
No need to run model1.py if predict output.
Please backup the existed method 1 models before running model1.py! 
'''

# Create Dir to same model1 (if not exist)
path = './model/model1'
if not os.path.exists(path):
    os.makedirs(path)


############ Load Training Data set ############

inputs, output = loadData()
sys.stdout.write("10%\r")

############ Preprocess of Sample Data set ############

# Feature data
output_number = output[:][0]

# Type of first sequence
input_new = pd.concat([inputs, output[:][0]], axis=1)
output_seq = output[:][1]
output_seq_new = output_seq.str.split(':', expand=True)
output_new = output_seq_new[[0]]

# Divide Train set and Output set
output_seq_split = output_seq.str.split(':', expand=True)
data_seq1 = output_seq_split[1].tolist()
sys.stdout.write("20%\r")

# Preprocess - attributes of the first sequence
# Find the regression number
seq1 = find(data_seq1)
data_sample = input_new
data_sample = data_sample.values.tolist()

# Type C sequence
input_c = []
output_c = []
# Type D sequence
input_d = []
output_d = []
# Valid
split_buffer = []

data_output_seqtype = output_seq_new[0].tolist()

for i in range(0, len(data_sample)):
    if data_output_seqtype[i] == "C":
        input_c.append(data_sample[i])
        output_c.append(seq1[i])
    elif data_output_seqtype[i] == "D":
        input_d.append(data_sample[i])
        output_d.append(seq1[i])
    else:
        split_buffer.append(seq1[i])

if split_buffer:
    print("Error - Process Training Data")
else:
    sys.stdout.write("30%\r")

# Preprocess - attributes of the second sequence
output_2seqtype = []
for i in data_seq1:
    a = re.sub('[^A-Z]+', '', i)
    output_2seqtype.append(a)

input_new_1 = []
for i in range(0, len(data_sample)):
    tem = data_sample[i] + seq1[i]
    input_new_1.append(tem)

# Find the regression number
q = output_seq_split.fillna("'(0,0),0'")
data_seq2 = q[2].tolist()
seq2 = find(data_seq2)

# Type D sequence
input_2d = []
output_2d = []

# Type C sequence
input_2c = []
output_2c = []

for i in range(0, len(input_new_1)):
    if output_2seqtype[i] == "D":
        input_2d.append(input_new_1[i])
        output_2d.append(seq2[i])
    elif output_2seqtype[i] == "C":
        input_2c.append(input_new_1[i])
        output_2c.append(seq2[i])
sys.stdout.write("40%\r")

# Preprocess - attributes of the third sequence
output_3seqtype = []
for i in data_seq2:
    a = re.sub('[^A-Z]+', '', i)
    output_3seqtype.append(a)

for i in range(len(output_3seqtype)):
    if output_3seqtype[i] is None:
        output_3seqtype[i] = '0'

data_seq3 = q[3].tolist()
seq3 = find(data_seq3)

input_new_12 = []
for i in range(0, len(data_sample)):
    tem = data_sample[i] + seq1[i] + seq2[i]
    input_new_12.append(tem)

# Type D sequence
input_3d = []
output_3d = []
for i in range(0, len(input_new_12)):
    if output_3seqtype[i] == "D":
        input_3d.append(input_new_12[i])
        output_3d.append(seq3[i])
sys.stdout.write("50%\r")

############ Prediction models ############

# 1.Feature model
model_number = DecisionTreeRegressor(max_depth=30)
model_number.fit(inputs, output_number)
sys.stdout.write("60%\r")
joblib.dump(filename=path + '/number.model', value=model_number)

# 2.Type model of 1st sequence
model_1_seqtype = DecisionTreeClassifier(max_depth=50)
model_1_seqtype.fit(input_new, output_new)
sys.stdout.write("65%\r")
joblib.dump(filename=path + '/model_1_seqtype.model', value=model_1_seqtype)

# 3.C sequence model of 1st sequence
model_1c_seqdata = DecisionTreeRegressor(max_depth=50)
model_1c_seqdata.fit(input_c, output_c)
sys.stdout.write("70%\r")
joblib.dump(filename=path + '/model_1c_seqdata.model', value=model_1c_seqdata)

# 4.D sequence model of 1st sequence
model_1d_seqdata = DecisionTreeRegressor(max_depth=30)
model_1d_seqdata.fit(input_d, output_d)
sys.stdout.write("75%\r")
joblib.dump(filename=path + '/model_1d_seqdata.model', value=model_1d_seqdata)

# 5.Type model of 2nd sequence
model_2_seqtype = DecisionTreeClassifier(max_depth=50)
model_2_seqtype.fit(input_new, output_2seqtype)
sys.stdout.write("80%\r")
joblib.dump(filename=path + '/model_2_seqtype.model', value=model_2_seqtype)

# 6.C sequence model of 2nd sequence
model_2cseq = DecisionTreeRegressor(max_depth=30)
model_2cseq.fit(input_2c, output_2c)
sys.stdout.write("85%\r")
joblib.dump(filename=path + '/model_2cseq.model', value=model_2cseq)

# 7.D sequence model of 2nd sequence
model_2dseq = DecisionTreeRegressor(max_depth=30)
model_2dseq.fit(input_2d, output_2d)
sys.stdout.write("90%\r")
joblib.dump(filename=path + '/model_2dseq.model', value=model_2dseq)

# 8.Type model of 3rd sequence
model_3_seqtype = DecisionTreeClassifier(max_depth=50)
model_3_seqtype.fit(input_new, output_3seqtype)
sys.stdout.write("95%\r")
joblib.dump(filename=path + '/model_3_seqtype.model', value=model_3_seqtype)

# 9.D sequence model of 3rd sequence
model_3d_seqdata = DecisionTreeRegressor(max_depth=30)
model_3d_seqdata.fit(input_3d, output_3d)
sys.stdout.write("100%\r")
joblib.dump(filename=path + '/model_3d_seqdata.model', value=model_3d_seqdata)
