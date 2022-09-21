# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model.layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability

class SpGAT(nn.Module):
    def __init__(self, in_dim, dropout, alpha):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attention = SpGraphAttentionLayer(in_dim, dropout=dropout, alpha=alpha, concat=True)

        self.W_r = nn.Parameter(torch.zeros(size=(in_dim, in_dim)))
        nn.init.xavier_uniform_(self.W_r, gain=nn.init.calculate_gain('relu'))

        # W matrix to convert h_input to h_output dimension

    def forward(self, ent_embedding, rel_embedding, edge_list, rel_list):
        input_ent_embedding = ent_embedding
        input_rel_embedding = rel_embedding

        if (CUDA):
            edge_list = edge_list.cuda()
            rel_list = rel_list.cuda()

        out_ent_embedding  = self.attention(input_ent_embedding,  input_rel_embedding, edge_list, rel_list)
        out_ent_embedding  = F.relu(self.dropout_layer(out_ent_embedding))

        out_rel_embedding = F.relu(input_rel_embedding.mm(self.W_r)) + input_rel_embedding

        return out_ent_embedding, out_rel_embedding

class SpKBGATModified(nn.Module):
    def __init__(self, ent_num, rel_num, in_dim, drop_GAT, alpha):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = ent_num
        self.entity_in_dim = in_dim

        # Properties of Relations
        self.num_relation = rel_num
        self.relation_dim = in_dim

        self.in_dim = in_dim

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.sparse_gat = SpGAT(self.in_dim, self.drop_GAT, self.alpha, )

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.in_dim)))
        nn.init.xavier_uniform_(self.W_entities, gain=nn.init.calculate_gain('relu'))

    def forward(self, adj, entity_embeddings, relation_embeddings):
        # getting edge list
        edge_list = adj[0] #头实体和对应的尾实体
        rel_list = adj[1] #对应的关系

        if(CUDA):
            edge_list = edge_list.cuda()
            rel_list = rel_list.cuda()

        out_entity, out_relation  = self.sparse_gat.forward(entity_embeddings, relation_embeddings, edge_list, rel_list)

        # mask_indices1 = torch.unique(batch_inputs[:, 0]).cuda()
        # mask_indices2 = torch.unique(batch_inputs[:, 2]).cuda()
        # mask = torch.zeros(entity_embeddings.shape[0]).cuda()
        # mask[mask_indices1] = 1.0
        # mask[mask_indices2] = 1.0

        entities_upgraded = entity_embeddings.mm(self.W_entities)
        #out_entity = entities_upgraded + mask.unsqueeze(-1).expand_as(out_entity) * out_entity
        out_entity = entities_upgraded + out_entity

        out_entity = F.normalize(out_entity, p=2, dim=1)
        out_relation = F.normalize(out_relation, p=2, dim=1)

        return out_entity, out_relation

class SpKBGATConvOnly(nn.Module):
    def __init__(self, ent_num, rel_num, in_dim, drop_conv, out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = ent_num
        self.entity_in_dim = in_dim

        # Properties of Relations
        self.num_relation = rel_num
        self.relation_dim = in_dim

        self.in_dim = in_dim

        self.out_channels = out_channels
        self.drop_conv = drop_conv

        self.final_entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.in_dim))
        self.final_relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.in_dim))

        self.convKB = ConvKB(self.in_dim, self.out_channels, self.drop_conv)

    def forward(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

class ConvE(nn.Module):
    def __init__(self, args, ent_num, rel_num):
        super(ConvE, self).__init__()

        self.ent_num = ent_num
        self.rel_num = rel_num

        self.embedding_dim = args.embedding_dim
        self.embedding_dim1 = args.embedding_dim1
        self.embedding_dim2 = args.embedding_dim // args.embedding_dim1

        self.final_entity_embeddings = nn.Parameter(torch.randn(self.ent_num, self.embedding_dim), requires_grad=True)
        self.final_relation_embeddings = nn.Parameter(torch.randn(self.rel_num, self.embedding_dim), requires_grad=True)

        self.relu = nn.ReLU()

        self.inp_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop, inplace=False)
        self.feature_drop = torch.nn.Dropout2d(args.feat_drop, inplace=False)

        self.conv = nn.Conv2d(1, 32, 4, padding=1)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm1d(args.embedding_dim)
        self.bn_conv = nn.BatchNorm2d(32)

        self.fc = nn.Linear(args.hidden_dim, args.embedding_dim)
        self.loss = nn.CrossEntropyLoss()

    # 解码器函数
    def circonv(self, input_matrix):
        x = input_matrix[:, :, :, :1]
        output_matrix = torch.cat((input_matrix, x), dim=3)
        y = output_matrix[:, :, :1, :]
        temp_matrix = output_matrix.clone()
        output_matrix = torch.cat((temp_matrix, y), dim=2)

        return output_matrix

    def forward(self, quadruples, pred):
        if pred == 'sub':  # 头实体预测
            tailList = quadruples[:, 2]
            relList = quadruples[:, 1]
            tail_embedding = self.final_entity_embeddings[tailList].to('cuda')
            rel_embedding = self.final_relation_embeddings[relList].to('cuda')

            head_embedding = self.final_entity_embeddings.to('cuda')

            tail_embed = tail_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            rel_embed = rel_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((tail_embed, rel_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, head_embedding.transpose(1, 0))

        elif pred == 'obj':  # 尾实体预测
            headList = quadruples[:, 0]
            relList = quadruples[:, 1]
            head_embedding = self.final_entity_embeddings[headList].to('cuda')
            rel_embedding = self.final_relation_embeddings[relList].to('cuda')

            tail_embedding = self.final_entity_embeddings.to('cuda')

            head_embed = head_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            rel_embed = rel_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((head_embed, rel_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, tail_embedding.transpose(1, 0))

        else:  # 关系预测
            headList = quadruples[:, 0]
            tailList = quadruples[:, 2]
            head_embedding = self.final_entity_embeddings[headList].to('cuda')
            tail_embedding = self.final_entity_embeddings[tailList].to('cuda')

            rel_embedding = self.final_relation_embeddings.to('cuda')

            head_embed = head_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            tail_embed = tail_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((head_embed, tail_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, rel_embedding.transpose(1, 0))

        return score