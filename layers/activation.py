import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, X):
        return X


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            # act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif issubclass(act_name, nn.Module):
            act_layer = act_name()
        else:
            raise NotImplementedError

        return act_layer