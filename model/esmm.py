"""
@Author: 1365677361@qq.com

@info:
"""

from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, concat_fun
from layers.core import DNN


class ESMM(BaseTower):
    """Entire Space Multi-task Model (ESMM)"""
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(128, 64), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(ESMM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device, gpus=gpus)

        self.ctr_dnn = DNN(-1, dnn_hidden_units,activation=dnn_activation, l2_reg=l2_reg_dnn,
                           dropout_rate=dnn_dropout,use_bn=dnn_use_bn, init_std=init_std, device=device)

        self.cvr_dnn = DNN(-1, dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def forward(self, inputs):
        user_sparse_embedding_list, user_dense_value_list = \
            self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

        item_sparse_embedding_list, item_dense_value_list = \
            self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

        item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

        dnn_input = concat_fun([user_dnn_input, item_dnn_input])

        ctr = self.ctr_dnn(dnn_input)
        cvr = self.cvr_dnn(dnn_input)

        ctcvr = ctr * cvr

        # TODO
        return ctcvr, ctr
