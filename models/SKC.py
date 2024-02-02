import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SKConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        M=2,
        groups=1,
        r=8,
        L=4,
        bias0=False,
        bias1=True,
        bias2=False,
        num_groups=10,
    ) -> None:
        super().__init__()

        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=groups,
                        bias=bias0,
                    ),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.ReLU(),
                )
            )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(nn.Conv1d(out_channels, d, 1, bias=bias1), nn.ReLU())
        self.fc2 = nn.Conv1d(d, out_channels * M, kernel_size=1, bias=bias2)
        self.softmax = nn.Softmax(dim=1)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(inputs))
        output = torch.stack(output, dim=1)
        U = output.sum(dim=1)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        return torch.sum(torch.mul(a_b, output), dim=1)
