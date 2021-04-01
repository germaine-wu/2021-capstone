
import pandas as pd
import numpy as np
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputRegressor

# 1.read training input
data_sample = pd.read_csv("data/input.txt", sep=',', header=None)
data_sample = data_sample.values.tolist()

# 2.read training output

# 2.1 get the first number
output = pd.read_csv("data/output.txt", sep='|', header=None)
data_output_flag = output[:][0].tolist()

# 2.2 get the type of first sequence(C or D)
output = output[:][1]
data_output_split = output.str.split(':', expand=True)
data_seq_type = data_output_split[0].tolist()

# 2.3 get the regression number
#     (first column for now)


def get_regression(data, num):
    data_seq = data[num].tolist()
    sequence = []
    # find out the content in ()
    p1 = re.compile(r'[(](.*?)[)]')
    for seq in data_seq:
        temp = re.findall(p1, seq)
        temp1 = ''.join(temp)
        temp2 = temp1.split(',')
        temp2 = list(map(eval, temp2))
        sequence.append(temp2)
    return sequence


seq1 = get_regression(data_output_split, 1)

# Divide Train set and Output set
# Type: ONE C item
input_one = []
output_one = []
# Type C sequence
input_c = []
output_c = []
# Type D sequence
input_d = []
output_d = []
# Valid
split_buffer = []

for i in range(0, len(data_sample)):
    if data_output_flag[i] == 0:
        input_one.append(data_sample[i])
        output_one.append(seq1[i])
    elif data_output_flag[i] != 0:
        if data_seq_type[i] == "C":
            input_c.append(data_sample[i])
            output_c.append(seq1[i])
        elif data_seq_type[i] == "D":
            input_d.append(data_sample[i])
            output_d.append(seq1[i])
        else:
            split_buffer.append(seq1[i])
    else:
        split_buffer.append(seq1[i])


# First Number Model
class FlagModel(object):
    def __init__(self):
        self.inputs = data_sample
        self.output = data_output_flag

    def prediction(self, inputs):
        flag_model = DecisionTreeRegressor(max_depth=26)
        flag_model.fit(self.inputs, self.output)
        pred = flag_model.predict(inputs)
        return pred


# Seq_Type Prediction Model
class SeqTypeModel(object):
    def __init__(self):
        self.input = data_sample
        self.output = data_seq_type

    def prediction(self, inputs):
        seq_type_model = DecisionTreeClassifier()
        seq_type_model.fit(self.input, self.output)
        pred = seq_type_model.predict(inputs)
        return pred


# Type One Prediction Model
class ZeroModel(object):
    def __init__(self):
        self.input = input_one
        self.output = output_one

    def prediction(self, inputs):
        zero_model = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))
        zero_model.fit(self.input, self.output)
        pred = zero_model.predict(inputs)
        pred = np.array(pred).astype(dtype=float).tolist()
        return pred


class RNNModel(object):
    # Type two model, finish later
    def __init__(self):
        self.input = input_c
        self.output = output_c
