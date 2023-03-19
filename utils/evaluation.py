import numpy as np 
import os
from sklearn.metrics import roc_auc_score, f1_score



def evaluation(result_list, num_cls):
    total = len(result_list)

    # predict = np.zeros((num_cls, total), dtype=np.float64)
    # target = np.zeros((num_cls, total), dtype=np.float64)

    good = 0
    for i in range(total):
        pred = result_list[i]["pred"]
        label = result_list[i]["label"]
        # predict[pred][i] = 1
        # target[label][i] = 1
        if pred == label: good += 1
    # print(predict)
    # print(target)
    acc = good / total
    # auc = 0
    # for j in range(num_cls):
    #     auc += roc_auc_score(target[j], predict[j])
    # auc = auc / num_cls

    return acc




