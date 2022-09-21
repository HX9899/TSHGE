# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

CUDA = torch.cuda.is_available()

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, dropout, alpha, concat=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.alpha = alpha
        self.concat = concat

        self.W_e1 = nn.Parameter(torch.zeros(size=(in_features, 3 * in_features)))
        nn.init.xavier_normal_(self.W_e1, gain=1.414)
        self.W_e2 = nn.Parameter(torch.zeros(size=(1, in_features)))
        nn.init.xavier_normal_(self.W_e2, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, ent_embedding, rel_embedding, edge, rel):
        N = ent_embedding.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat(
            (ent_embedding[edge[0, :], :], rel_embedding[rel, :], ent_embedding[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.W_e1.mm(edge_h)
        # edge_m: D * E  #E是三元组数量，D是输出向量纬度

        # to be checked later
        powers = -self.leakyrelu(self.W_e2.mm(edge_m).squeeze())

        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(    #目前edge是N*1
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.in_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.relu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.in_features) + ')'

class ConvKB(nn.Module):
    def __init__(self, in_dim, out_channels, drop_conv):
        super().__init__()

        self.conv_layer = nn.Conv2d(1, out_channels, (1, 3))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_conv)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((in_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output