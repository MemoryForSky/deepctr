import torch
import torch.nn as nn
from model.base_model import BaseModel
from layers.interaction import CrossNet
from layers.core import DNN
from preprocessing.inputs import combined_dnn_input


class DCN(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=(256, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_cross=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None):
        super(DCN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        # Deep
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       use_bn=dnn_use_bn, init_std=init_std, device=device)

        # Linear
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)

        # Cross
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns),
                                 layer_num=cross_num, parameterization=cross_parameterization, device=device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)
        self.to(device)

    def forward(self, inputs):
        logit = self.linear_model(inputs)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(inputs, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((deep_out, cross_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:
            pass
        y_pred = self.out(logit)
        return y_pred
