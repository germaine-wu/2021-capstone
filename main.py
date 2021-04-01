import warnings
import numpy as np
import pandas as pd

from models import FlagModel
from models import SeqTypeModel
from models import ZeroModel


# read input
def read_data(filename):
    data_sample = pd.read_csv(filename, sep=',', header=None)
    data_sample = data_sample.values.tolist()

    return data_sample


def write_data(filename, output):
    fw = open(filename, 'w')
    for item in output:
        fw.write(item+"\n")


# change number list to str list
def convert_list(lists):
    str1 = ''
    for item in lists:
        str1 += str(item)
    return str1


def train(inputs):

    result = []

    flag_model = FlagModel()
    seq_model = SeqTypeModel()
    zero_model = ZeroModel()
    flag = flag_model.prediction(inputs)
    seq = seq_model.prediction(inputs)

    for ind in range(len(flag)):
        print_str = str(int(flag[ind])) + "|" + seq[ind]
        if flag[ind] == 0:
            zero_input = np.array(inputs[ind]).reshape(1, -1)
            zero_seq = zero_model.prediction(zero_input)
            con_sir = convert_list(zero_seq)
            str1 = print_str + con_sir
            # replace "[]" to "()"
            str11 = str1.replace('[', '(')
            str22 = str11.replace(']', ')')
            result.append(str22)
        elif flag[ind] != 0:
            if seq[ind] == 'C':
                str2 = print_str + " (Type2 - Later)"
                result.append(str2)
            elif seq[ind] == 'D':
                str3 = print_str + " (Type3 - Later)"
                result.append(str3)
        else:
            result.append("Resolve later.")
    return result


if __name__ == '__main__':
    read_file_name = "input_test.txt"
    write_file_name = "output_test.txt"

    # Read - Pred - Write
    data_input = read_data(read_file_name)
    pre = train(data_input)
    write_data(write_file_name, pre)