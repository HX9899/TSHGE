# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import os

def sample(batch_indices, all_indices, pred, ent_num, rel_num):
    batch_num = len(batch_indices)
    batch_indices_with_neg = np.tile(batch_indices, (2, 1))

    for i in range(batch_num):
        if pred == 'sub':
            while(batch_indices_with_neg[batch_num+i, 0], batch_indices_with_neg[batch_num+i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 0] = np.random.randint(0, ent_num)

        elif pred == 'obj':
            while(batch_indices_with_neg[batch_num+i, 0], batch_indices_with_neg[batch_num+i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 2] = np.random.randint(0, ent_num)

        else:
            while (batch_indices_with_neg[batch_num + i, 0], batch_indices_with_neg[batch_num + i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 1] = np.random.randint(0, rel_num)

    return batch_indices_with_neg

def construct_snap(test_quadruples, final_score, topK, pred):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_quadruples = []

    rows, cols, data = [], [], []
    
    if pred == 'sub':
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                t, r = test_quadruples[_][2], test_quadruples[_][1]
                predict_quadruples.append((index, r, t))
                
    elif pred == 'obj':
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                h, r = test_quadruples[_][0], test_quadruples[_][1]
                predict_quadruples.append((h, r, index))
                
    else:
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                h, t = test_quadruples[_][0], test_quadruples[_][2]
                predict_quadruples.append((h, index, t))

    for quadruple in predict_quadruples:
        rows.append(quadruple[0])
        cols.append(quadruple[2])
        data.append(quadruple[1])
        rows.append(quadruple[2])
        cols.append(quadruple[0])
        data.append(quadruple[1])

    return rows, cols, data
