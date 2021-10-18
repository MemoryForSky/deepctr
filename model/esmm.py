"""
@Author: 1365677361@qq.com

@info:
"""

from model.base_tower import BaseTower


class ESMM(BaseTower):
    """Entire Space Multi-task Model (ESMM)"""
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(ESMM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device, gpus=gpus)
        pass

    def forward(self, inputs):
        pass
