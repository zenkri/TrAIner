import json
import numpy as np
import matplotlib.pyplot as plt
import ast


def json2data(jsondata):
    train_data = []
    labels = []

    for row_index in jsondata:
        try:
            row = ast.literal_eval(jsondata[row_index])
            train_data.append([row['ypr']['x'], row['ypr']['y'], row['ypr']['z'],
                               row['acc']['x'], row['acc']['y'], row['acc']['z'],
                               row['apert']['w'], row['apert']['x'], row['apert']['y'], row['apert']['z']])

            labels.append([list(row['lab'])[0]])
        except Exception as e:
            print(e)

    return np.array(train_data, dtype=float), np.array(labels, dtype=float)


if __name__=="__main__":

    dir_to_file = '/home/oussama/Desktop/TrAIner/data/stopngo.json'
    with open(dir_to_file) as jsonFile:
        try:
            data = json.load(jsonFile)

        except Exception as e:
            print(e)

    train_data, labels = json2data(data)
    print(' ')