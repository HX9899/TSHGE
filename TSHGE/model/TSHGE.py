# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from utils.Args import args
import time
import math

from model.models import SpKBGATModified

CUDA = torch.cuda.is_available()

class tshge(torch.nn.Module):
    def __init__(self, args, ent_num, rel_num):

        super(tshge, self).__init__()

        # 编码器
        self.embedding_dim = args.embedding_dim

        self.ent_num = ent_num
        self.rel_num = rel_num

        # 获取实体、关系、时间戳的嵌入权重矩阵
        self.entity_init_embeds = nn.Parameter(torch.Tensor(self.ent_num, self.embedding_dim), requires_grad=True)
        self.relation_init_embeds = nn.Parameter(torch.Tensor(self.rel_num, self.embedding_dim), requires_grad=True)
        self.timestamp_init_embeds = nn.Parameter(torch.Tensor(1, self.embedding_dim), requires_grad=True)

        self.final_ent_embeds = nn.Parameter(torch.Tensor(self.ent_num, self.embedding_dim), requires_grad=True)
        self.final_rel_embeds = nn.Parameter(torch.Tensor(self.rel_num, self.embedding_dim), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()  # inplace=False)
        self.tanh = nn.Tanh()

        self.drop_ta = nn.Dropout(args.dropout_ta)

        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.gat = SpKBGATModified(self.ent_num, self.rel_num, self.embedding_dim, args.dropout_ta, args.alpha)

        self.w_t_fea = nn.Parameter(torch.zeros(size=(self.embedding_dim, self.embedding_dim)))
        self.w_t_ent = nn.Parameter(torch.zeros(size=(self.ent_num, 1)))

        self.w_t_fea_r = nn.Parameter(torch.zeros(size=(self.embedding_dim, self.embedding_dim)))
        self.w_t_rel = nn.Parameter(torch.zeros(size=(self.rel_num, 1)))

        self.loss = nn.MarginRankingLoss(margin=args.margin)

        self.reg_para = args.reg_para

        self.reset_parameters()

    # 编码器函数
    def reset_parameters(self):  # 初始化实体、关系、时间戳的嵌入权重矩阵
        nn.init.xavier_uniform_(self.entity_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.timestamp_init_embeds, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.final_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.final_rel_embeds, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.w_t_fea, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_t_ent, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.w_t_fea_r, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_t_rel, gain=nn.init.calculate_gain('relu'))

    def get_init_timesatmp(self, timeList):  # 获取所有时间戳的嵌入矩阵，输入时间戳是时间戳列表
        init_timestamps = torch.Tensor(len(timeList), self.embedding_dim)

        for i in range(len(timeList)):
            init_timestamps[i] = torch.Tensor(
                self.timestamp_init_embeds.cpu().detach().numpy().reshape(self.embedding_dim)) * (timeList[i] // args.time_interval + 1)
            # self.timestamp_init_embeds.cpu().detach().numpy().reshape(self.embedding_dim)) * (i + 1)

        init_timestamps = init_timestamps.to('cuda')

        return init_timestamps

    def get_temporal_entity_And_relation_Embed_with_t(self, timeList, adjList):
        init_timestamps = self.get_init_timesatmp(timeList)

        # init_entity = self.entity_init_embeds.clone().to('cuda')  # 获取每个时间戳下的所有实体的初始嵌入矩阵
        # init_relation = self.relation_init_embeds.clone().to('cuda')  # 获取每个时间戳下的所有实体的初始嵌入矩阵

        ent_mat_t = torch.ones(len(timeList), self.ent_num, self.embedding_dim).cuda()
        rel_mat_t = torch.ones(len(timeList), self.rel_num, self.embedding_dim).cuda()

        for i in range(len(timeList)):
            timestamp = timeList[i] // args.time_interval

            # 时间感知注意力
            entity_embed_t = torch.matmul(init_timestamps[i], self.w_t_fea).float()
            impact_factor_e = F.softmax(entity_embed_t, dim=-1)
            impact_factor_e_mat = impact_factor_e.repeat(self.ent_num, 1)
            impact_factor_e_mat = self.w_t_ent * impact_factor_e_mat

            ent_embedding = self.relu(self.drop_ta(
                torch.mul(self.entity_init_embeds * (timestamp + 1), impact_factor_e_mat))) + self.entity_init_embeds

            relation_embed_t = torch.matmul(init_timestamps[i], self.w_t_fea_r).float()
            impact_factor_r = F.softmax(relation_embed_t, dim=-1)
            impact_factor_r_mat = impact_factor_r.repeat(self.rel_num, 1)
            impact_factor_r_mat = self.w_t_rel * impact_factor_r_mat

            rel_embedding = self.relu(self.drop_ta(
                torch.mul(self.relation_init_embeds * (timestamp + 1), impact_factor_r_mat))) + self.relation_init_embeds

            #邻域gat
            if i < len(timeList)-1:
                adj = adjList[i]
                adj_indices = torch.LongTensor([adj[0], adj[1]])
                adj_values = torch.LongTensor(adj[2])
                adj_mat = (adj_indices, adj_values)
                out_ent_embedding, out_rel_embedding = self.gat.forward(adj_mat, ent_embedding, rel_embedding)
                ent_mat_t[i] = out_ent_embedding
                rel_mat_t[i] = out_rel_embedding
            
            else:
                ent_mat_t[i] = ent_embedding
                rel_mat_t[i] = rel_embedding

        #特征继承
        _, out_ent_embeds = self.gru(ent_mat_t)
        _, out_rel_embeds = self.gru(rel_mat_t)

        out_ent_embeds = torch.squeeze(out_ent_embeds, 0)
        out_rel_embeds = torch.squeeze(out_rel_embeds, 0)

        self.final_ent_embeds.weight = out_ent_embeds.data
        self.final_rel_embeds.weight = out_rel_embeds.data

        return out_ent_embeds, out_rel_embeds

    def forward(self, train_indices, ent_embeds, rel_embeds):
        len_pos_triples = int(train_indices.shape[0] / 2)

        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]

        pos_triples = pos_triples.repeat(1, 1)

        source_embeds =ent_embeds[pos_triples[:, 0]].cuda()
        relation_embeds = rel_embeds[pos_triples[:, 1]].cuda()
        tail_embeds = ent_embeds[pos_triples[:, 2]].cuda()

        x = source_embeds + relation_embeds - tail_embeds
        pos_norm = torch.norm(x, p=1, dim=1)

        source_embeds_2 = ent_embeds[neg_triples[:, 0]].cuda()
        relation_embeds_2 = rel_embeds[neg_triples[:, 1]].cuda()
        tail_embeds_2 = ent_embeds[neg_triples[:, 2]].cuda()

        x_2 = source_embeds_2 + relation_embeds_2 - tail_embeds_2
        neg_norm = torch.norm(x_2, p=1, dim=1)

        y = -torch.ones(len_pos_triples).cuda()

        loss = self.loss(pos_norm, neg_norm, y)

        return loss

    def regularization_loss(self):
        regularization_loss = torch.mean(self.entity_init_embeds.pow(2)) + torch.mean(
            self.relation_init_embeds.pow(2)) + torch.mean(self.timestamp_init_embeds.pow(2))
        return regularization_loss * self.reg_para
























































