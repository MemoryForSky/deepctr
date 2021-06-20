import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()

    def forward(self, inputs):
        ffm_input = inputs


class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.kernels = nn.Parameter(torch.Tensor(layer_num, in_features, 1))
        self.bias = nn.Parameter(torch.Tensor(layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # [B, D, 1]
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=[[1], [0]])  # [B, 1, 1]
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class AFMLayer(nn.Module):
    def __init__(self, in_features, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, device='cpu'):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features

        self.attention_W = nn.Parameter(torch.Tensor(embedding_size, self.attention_factor))

        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))

        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor, )

        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        self.dropout = nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q  # [B, len(row), embedding_size]

        bi_interaction = inner_product
        # [B, len(row), self.attention_factor]
        attention_temp = F.relu(torch.tensordot(bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)

        # [B, len(row), 1]
        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        # [B, 1, dmb_size]
        attention_output = torch.sum(self.normalized_att_score * bi_interaction, dim=1)

        attention_output = self.dropout(attention_output)

        # [B, 1, 1]
        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))

        return afm_out
