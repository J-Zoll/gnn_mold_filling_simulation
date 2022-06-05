from torch import nn


class MLP (nn.Module):
    def __init__(self, inp_size, hid_size, out_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
