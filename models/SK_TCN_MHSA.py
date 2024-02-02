from models.SKC import SKConv
from models.TCN import TemporalConvNet
import torch.nn as nn
from typing import Dict, Any
from models.MHSA import MultiHeadedSelfAttention
from models.MLPs import MLP


class classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_channels,
        dropout,
        kernel_size,
        L,
        MHSA_dim,
        r=10,
        num_linears=[64],
        dropout_linears=[0.2],
        mhsa_drop=0.1,
        seqlen=41,
        n_heads=3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SKConv(in_channels, num_channels[0], 1, 2, 1, r, L, False, True, False),
            TemporalConvNet(
                num_inputs=num_channels[0], num_channels=num_channels[1:], kernel_size=kernel_size, dropout=dropout
            ),
            MultiHeadedSelfAttention(num_channels[-1], MHSA_dim * n_heads, mhsa_drop, n_heads),
            nn.Flatten(),
        )
        input_linear = seqlen * num_channels[-1]
        self.net.append(MLP(input_linear, num_linears, dropout_linears, acti="hardswish"))

    def forward(self, X):
        return self.net(X.permute(0, 2, 1))

    def get_code(self, X):
        return self.net[0](X.permute(0, 2, 1))

    @staticmethod
    def get_model_params():
        model_params = dict(
            in_channels=5,
            MHSA_dim=11,
            n_heads=3,
            num_channels=[190, 100, 100, 100, 80, 33],
            num_linears=[700, 220],
            kernel_size=4,
            mhsa_drop=0.1,
            L=78,
            r=6,
            seqlen=201,
            dropout=0.1,
            dropout_linears=[0.4, 0.2],
        )
        return model_params

    @staticmethod
    def get_hparams() -> Dict[str, Any]:
        hparams = dict(batchsize=64, lr=1e-4, patience=100, monitor="valid_S", name="STM-ac4C", max_epochs=400)
        return hparams
