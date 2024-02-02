import torch.nn as nn
from typing import Literal
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        num_linears,
        dropout_linears,
        acti: Literal["hardswish", "relu", "elu", "gelu", "selu", "mish"],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        input_linear = in_features
        if acti == "hardswish":
            self.acti = nn.Hardswish
        elif acti == "relu":
            self.acti = nn.ReLU
        elif acti == "elu":
            self.acti = nn.ELU
        elif acti == "gelu":
            self.acti = nn.GELU
        elif acti == "selu":
            self.acti = nn.GELU
        elif acti == "mish":
            self.acti = nn.Mish
        else:
            raise NotImplementedError()

        for nl, dr in zip(num_linears, dropout_linears):
            self.net.append(nn.Dropout(dr))
            self.net.append(nn.Linear(input_linear, nl))
            self.net.append(self.acti())
            input_linear = nl
        self.net.append(nn.Linear(input_linear, 1))
        self.net.append(nn.Flatten(start_dim=0))
        # self.apply(self.init_func)

    @staticmethod
    def init_func(m):
        if isinstance(m, nn.Linear):
            # m.weight.data.normal_(0, 0.01)
            init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, X):
        return self.net(X)
