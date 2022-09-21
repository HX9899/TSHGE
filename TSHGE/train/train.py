# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch import optim
import math
import numpy as np
import scipy.sparse as sp
from utils.Args import args
from utils.dataset import Dataset
from model.TSHGE import tshge
from model.models import ConvE
from utils.measure import *
from utils.util import *
import random
import time
from tqdm import tqdm
import warnings
import gc

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True

warnings.filterwarnings(action='ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 数据集加载
data_loader = Dataset(args.dataset)

quadrupleList_t_train, timestampsList_train, adj_t_train = data_loader.get_quadruple_with_t('train.txt')
quadrupleList_t_valid, timestampsList_valid, adj_t_valid = data_loader.get_quadruple_with_t('valid.txt')
quadrupleList_t_test, timestampsList_test, adj_t_test = data_loader.get_quadruple_with_t('test.txt')

all_quadrupleList_t = quadrupleList_t_train + quadrupleList_t_valid + quadrupleList_t_test

trained_quadrupleList_t = quadrupleList_t_train + quadrupleList_t_valid
trained_timestampsList = timestampsList_train + timestampsList_valid
trained_adj_t = adj_t_train + adj_t_valid

print('dataset dataload succeed')

all_entityList, all_relationList = data_loader.get_entityAndRlationList(all_quadrupleList_t)
all_entityList.sort()
all_relationList.sort()
ent_num = max(all_entityList) + 1
rel_num = max(all_relationList) + 1

print('dataset process succeed')

all_quadrupleList_t = []

batch_size_conv = args.batch_size_conv
his_length = args.train_history_length
test_his_length = args.test_history_length

history_timeList = timestampsList_valid[-test_his_length:]
history_adjList = adj_t_valid[-test_his_length:]

all_quadrupleList = []
all_timestampsList = []

if args.pred == 'sub':
    mkdirs('./results/sub/{}/th-gat'.format(args.dataset))
    mkdirs('./results/sub/{}/conve'.format(args.dataset))
    model_state_file1 = "./results/sub/{}/th-gat/model_state.pth".format(args.dataset)
    model_state_file2 = "./results/sub/{}/conve/model_state.pth".format(args.dataset)
elif args.pred == 'obj':
    mkdirs('./results/obj/{}/th-gat'.format(args.dataset))
    mkdirs('./results/obj/{}/conve_mul'.format(args.dataset))
    model_state_file1 = "./results/obj/{}/th-gat/model_state.pth".format(args.dataset)
    model_state_file2 = "./results/obj/{}/conve_mul/model_state.pth".format(args.dataset)
else:
    mkdirs('./results/rel/{}/th-gat'.format(args.dataset))
    mkdirs('./results/rel/{}/conve'.format(args.dataset))
    model_state_file1 = "./results/rel/{}/th-gat/model_state.pth".format(args.dataset)
    model_state_file2 = "./results/rel/{}/conve/model_state.pth".format(args.dataset)

def train_gat(args):
    print('start th-gat train')
    # 模型加载
    model = tshge(args, ent_num, rel_num)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_gat, amsgrad=True)

    if os.path.exists(model_state_file1):
        checkpoint = torch.load(model_state_file1)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # print(optimizer.param_groups[0]['lr'])
        # optimizer.param_groups[0]['lr'] = 0.0001
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.n_epochs_gat)):
        train_loss = 0

        for time_idx in tqdm(range(len(trained_timestampsList))):
            model.train()

            train_quadruple_list = trained_quadrupleList_t[time_idx]

            batch_size = len(train_quadruple_list)

            if time_idx > 0:
                if time_idx >= his_length:
                    timeList = trained_timestampsList[time_idx - his_length: time_idx + 1]
                    adjList = trained_adj_t[time_idx - his_length: time_idx]
                else:
                    timeList = trained_timestampsList[: time_idx + 1]
                    adjList = trained_adj_t[: time_idx]
            else:
                timeList = [trained_timestampsList[time_idx]]
                adjList = []

            train_quadruple = np.array(train_quadruple_list)
            n_batch = (train_quadruple.shape[0] + batch_size - 1) // batch_size

            for idx in range(n_batch):
                out_ent_embeds, out_rel_embeds = model.get_temporal_entity_And_relation_Embed_with_t(timeList, adjList)
                out_ent_embeds = out_ent_embeds.cuda()
                out_rel_embeds = out_rel_embeds.cuda()

                batch_start = idx * batch_size
                batch_end = min(train_quadruple.shape[0], (idx + 1) * batch_size)

                train_batch_quadruple = train_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                train_batch_quadruple_with_neg = sample(train_batch_quadruple, train_quadruple_list, args.pred, ent_num, rel_num)
                train_batch_quadruple_with_neg = torch.LongTensor(train_batch_quadruple_with_neg).cuda()

                loss1 = model.forward(train_batch_quadruple_with_neg, out_ent_embeds, out_rel_embeds)
                loss2 = model.regularization_loss()
                loss = loss1 + loss2
                train_loss += loss.item()

                optimizer.zero_grad()
                # for name, parms in model.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")
                loss.backward(retain_graph=True)
                optimizer.step()
                #print("=============更新之后===========")
                # for name, parms in model.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

        print('each_epoch_loss:', train_loss)

        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                   model_state_file1)

def train_conv(args):
    print('start conv train')
    # 模型加载

    count = 0
    best_mrr = 0
    best_hits1 = 0
    best_hits3 = 0
    best_hits10 = 0

    model_gat = th_gat(args, ent_num, rel_num)
    model = ConvE(args, ent_num, rel_num)
    model_gat.to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_conv, amsgrad=True)

    checkpoint = torch.load(model_state_file1)
    model_gat.load_state_dict(checkpoint['state_dict'], strict=False)
    model_gat.eval()

    if os.path.exists(model_state_file2):
        checkpoint = torch.load(model_state_file2)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # print(optimizer.param_groups[0]['lr'])
        # optimizer.param_groups[0]['lr'] = 0.0001
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.n_epochs_conv)):
        train_loss = 0

        for time_idx in tqdm(range(len(trained_timestampsList))):
            model.train()

            train_quadruple_list = trained_quadrupleList_t[time_idx]

            if time_idx > 0:
                if time_idx >= his_length:
                    timeList = trained_timestampsList[time_idx - his_length: time_idx + 1]
                    adjList = trained_adj_t[time_idx - his_length: time_idx]
                else:
                    timeList = trained_timestampsList[: time_idx + 1]
                    adjList = trained_adj_t[: time_idx]
            else:
                timeList = [trained_timestampsList[time_idx]]
                adjList = []

            train_quadruple = np.array(train_quadruple_list)
            n_batch = (train_quadruple.shape[0] + batch_size_conv - 1) // batch_size_conv

            model_gat.get_temporal_entity_And_relation_Embed_with_t(timeList, adjList)
            model.final_entity_embeddings.data = model_gat.final_ent_embeds.data
            model.final_relation_embeddings.data = model_gat.final_rel_embeds.data

            for idx in range(n_batch):
                batch_start = idx * batch_size_conv
                batch_end = min(train_quadruple.shape[0], (idx + 1) * batch_size_conv)

                train_batch_quadruple = train_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                if args.pred == 'sub':
                    train_targets = torch.Tensor(train_batch_quadruple[:, 0]).long().to('cuda')  # 生成每个batch的标签
                    train_scores = model.forward(train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets)

                elif args.pred == 'obj':
                    train_targets = torch.Tensor(train_batch_quadruple[:, 2]).long().to('cuda')
                    train_scores = model.forward(train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets)
                    # loss = model.loss(train_scores, train_targets)

                else:
                    train_targets = torch.Tensor(train_batch_quadruple[:, 1]).long().to('cuda')
                    train_scores = model.forward(train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets)

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

        print('each_epoch_loss:', train_loss)

        if epoch < args.valid_epoch:
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       model_state_file2)

        if epoch >= args.valid_epoch:
            model.eval()

            test_mrr, test_hits1, test_hits3, test_hits10 = 0, 0, 0, 0
            num = 0
            with torch.no_grad():
                if args.multi_step:
                    for time_idx in tqdm(range(len(timestampsList_test))):
                        test_quadruple_list = quadrupleList_t_test[time_idx]

                        num += len(test_quadruple_list)

                        timeList = history_timeList + [timestampsList_test[time_idx]]
                        adjList = history_adjList

                        test_quadruple = np.array(test_quadruple_list)
                        n_batch = (test_quadruple.shape[0] + batch_size_conv - 1) // batch_size_conv

                        model_gat.get_temporal_entity_And_relation_Embed_with_t(timeList, adjList)
                        model.final_entity_embeddings.data = model_gat.final_ent_embeds.data
                        model.final_relation_embeddings.data = model_gat.final_rel_embeds.data

                        rows, cols, data = [], [], []

                        for idx in range(n_batch):
                            batch_start = idx * batch_size_conv
                            batch_end = min(test_quadruple.shape[0], (idx + 1) * batch_size_conv)

                            test_batch_quadruple = test_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                            if args.pred == 'sub':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 0]).long().to(
                                    'cuda')  # 生成每个batch的标签
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            elif args.pred == 'obj':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 2]).long().to('cuda')
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            else:
                                test_targets = torch.Tensor(test_batch_quadruple[:, 1]).long().to('cuda')
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            temp_mrr, temp_hits1, temp_hits3, temp_hits10 = get_performanceIndex(test_scores, test_targets)
                            temp_rows, temp_cols, temp_data = construct_snap(test_batch_quadruple, test_scores, args.topk, args.pred)
                            rows  += temp_rows
                            cols  += temp_cols
                            data += temp_data

                            test_mrr += temp_mrr * len(test_batch_quadruple)
                            test_hits1 += temp_hits1 * len(test_batch_quadruple)
                            test_hits3 += temp_hits3 * len(test_batch_quadruple)
                            test_hits10 += temp_hits10 * len(test_batch_quadruple)

                        history_timeList.pop(0)
                        history_timeList.append(timestampsList_test[time_idx])

                        history_adjList.pop(0)
                        history_adjList.append((rows, cols, data))

                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                else:
                    for time_idx in tqdm(range(len(timestampsList_test))):
                        test_quadruple_list = quadrupleList_t_test[time_idx]

                        num += len(test_quadruple_list)

                        timeList = history_timeList + [timestampsList_test[time_idx]]
                        adjList = history_adjList

                        test_quadruple = np.array(test_quadruple_list)
                        n_batch = (test_quadruple.shape[0] + batch_size_conv - 1) // batch_size_conv

                        model_gat.get_temporal_entity_And_relation_Embed_with_t(timeList, adjList)
                        model.final_entity_embeddings.data = model_gat.final_ent_embeds.data
                        model.final_relation_embeddings.data = model_gat.final_rel_embeds.data

                        for idx in range(n_batch):
                            batch_start = idx * batch_size_conv
                            batch_end = min(test_quadruple.shape[0], (idx + 1) * batch_size_conv)

                            test_batch_quadruple = test_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                            if args.pred == 'sub':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 0]).long().to('cuda')  # 生成每个batch的标签
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            elif args.pred == 'obj':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 2]).long().to('cuda')
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            else:
                                test_targets = torch.Tensor(test_batch_quadruple[:, 1]).long().to('cuda')
                                test_scores = model.forward(test_batch_quadruple, args.pred)

                            temp_mrr, temp_hits1, temp_hits3, temp_hits10 = get_performanceIndex(test_scores, test_targets)

                            test_mrr += temp_mrr * len(test_batch_quadruple)
                            test_hits1 += temp_hits1 * len(test_batch_quadruple)
                            test_hits3 += temp_hits3 * len(test_batch_quadruple)
                            test_hits10 += temp_hits10 * len(test_batch_quadruple)

                        history_timeList.pop(0)
                        history_timeList.append(timestampsList_test[time_idx])

                        history_adjList.pop(0)
                        history_adjList.append(adj_t_test[time_idx])

                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()

                test_mrr = test_mrr / num
                test_hits1 = test_hits1 / num
                test_hits3 = test_hits3 / num
                test_hits10 = test_hits10 / num

            print("\n MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
                  format(test_mrr, test_hits1, test_hits3, test_hits10))

            if test_mrr > best_mrr:
                best_mrr = test_mrr

                count = 0

                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                           model_state_file2)
            else:
                count += 1

            if test_hits1 > best_hits1:
                best_hits1 = test_hits1
            if test_hits3 > best_hits3:
                best_hits3 = test_hits3
            if test_hits10 > best_hits10:
                best_hits10 = test_hits10

            print(
                "Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
                    format(epoch + 1, train_loss, best_mrr, best_hits1, best_hits3, best_hits10))

            if count == args.count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break

train_gat(args)
print('th_gat train done')

train_conv(args)
print('conve train done')

