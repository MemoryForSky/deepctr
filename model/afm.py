import torch
from model.base_model import BaseModel
from layers.interaction import FM, AFMLayer


class AFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_attention=True, attention_factor=8,
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', gpus=None):
        super(AFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        self.use_attention = use_attention

        if use_attention:
            self.fm = AFMLayer(self.embedding_size, attention_factor, l2_reg_att, afm_dropout,
                               seed, device)
            self.add_regularization_weight(self.fm.attention_W, l2=l2_reg_att)
        else:
            self.fm = FM()

        self.to(device)

    def forward(self, inputs):
        logit = self.linear_model(inputs)
        sparse_embedding_list, _ = self.input_from_feature_columns(inputs, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if len(sparse_embedding_list) > 0:
            if self.use_attention:
                logit += self.fm(sparse_embedding_list)
            else:
                logit += self.fm(torch.cat(sparse_embedding_list, dim=1))

        y_pred = self.out(logit)
        return y_pred
