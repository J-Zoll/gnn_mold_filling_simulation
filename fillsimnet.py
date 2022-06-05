"""This module specifies the filling simulation module."""

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from mlp import MLP


class FillSimNet(torch.nn.Module):
    def __init__(self, num_mp_layers):
        super().__init__()
        INPUT_SIZE = 2
        HIDDEN_SIZE = 64
        OUTPUT_SIZE = 1
        self.encoder = MLP(
            inp_size=INPUT_SIZE,
            hid_size=HIDDEN_SIZE,
            out_size=HIDDEN_SIZE
        )

        self.processor = nn.ModuleList(
            [GCNConv(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(num_mp_layers)]
        )

        self.decoder = MLP(
            inp_size=HIDDEN_SIZE,
            hid_size=HIDDEN_SIZE,
            out_size=OUTPUT_SIZE
        )

    def forward(self, data):
        x = data.x
        x = self.encoder(x)

        for layer in self.processor:
            x = layer(x, edge_index=data.edge_index, edge_weight=data.edge_weight)

        out = self.decoder(x)
        out = torch.sigmoid(out)
        return out