"""MLP binária: logits na saída; sigmoid somente na inferência."""

import torch
from torch import nn


class StrokeMLP(nn.Module):
    def __init__(self, n_features, hidden, dropout=0.0):
        super().__init__()
        layers = []
        width = n_features
        for next_width in hidden:
            linear = nn.Linear(width, next_width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = next_width
        output = nn.Linear(width, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def predict_scores(model, matrix):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(matrix, dtype=torch.float32))).numpy()
