import re
import sys
import pandas as pd

train_file1 = './Training Data/103_train.txt'
output_file1 = './Training Data/103_out.txt'
# 10^4 Data set
train_file2 = './Training Data/104_train.txt'
output_file2 = './Training Data/104_out.txt'

inputs1 = pd.read_csv(train_file1, sep=',', header=None)
inputs2 = pd.read_csv(train_file2, sep=',', header=None)
inputs = pd.concat([inputs1, inputs2], axis=0)
data_sample = inputs.values.tolist()
output1 = pd.read_csv(output_file1, sep='|', header=None)
output2 = pd.read_csv(output_file2, sep='|', header=None)
output = pd.concat([output1, output2], axis=0)

data_out = output[:][1]
split = data_out.str.split(':', expand=True)

print(len(data_sample))
print(len(split))
