import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, p, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_q = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.proj_k = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.proj_v = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.drop = nn.Dropout(p)

    def forward(self, X):
        """
        B C S -> B H W S
        """

        q, k, v = self.proj_q(X), self.proj_k(X), self.proj_v(X)
        q, k, v = [torch.unflatten(x, 1, (self.n_heads, -1)) for x in [q, k, v]]
        # (B H S W) @ (B H W S) -> (B H S S)
        scores = q.transpose(-2, -1) @ k / (k.size(-2) ** 0.5)
        scores = self.drop(F.softmax(scores, dim=-2))
        # (B H W S) @ (B H S S) -> (B H W S)
        h = (v @ scores).reshape(k.size(0), -1, k.size(-1))
        return h


class MHSA_origin(nn.Module):
    def __init__(self, emb_dim, num_heads, p) -> None:
        super().__init__()
        self.net = nn.MultiheadAttention(emb_dim, num_heads, p, batch_first=True)

    def forward(self, X):
        _X = X.permute(0, 2, 1)
        out, self.weight = self.net(_X, _X, _X)
        return out.permute(0, 2, 1)
