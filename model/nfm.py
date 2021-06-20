import torch
import torch.nn as nn
from model.base_model import BaseModel
from layers.interaction import BiInteractionPooling
from layers.core import DNN
from preprocessing.inputs import combined_dnn_input


class NFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024, bi_dropout=0, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None):
        super(NFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        # Deep
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       use_bn=dnn_use_bn, init_std=init_std, device=device)

        # Linear
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)

        self.to(device)

    def forward(self, inputs):
        linear_logit = self.linear_model(inputs)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(inputs, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        dnn_input = combined_dnn_input([bi_out], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        logit = linear_logit + dnn_logit

        y_pred = self.out(logit)
        return y_pred
