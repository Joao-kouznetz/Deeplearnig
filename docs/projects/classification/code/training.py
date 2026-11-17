"""Loop legível: forward, BCE, backward, Adam, validação e melhor checkpoint."""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from network import StrokeMLP, predict_scores
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_network(xt, yt, xv, yv, config, seed, protocol):
    seed_everything(seed)
    model = StrokeMLP(xt.shape[1], config["hidden"], config["dropout"])
    tx, ty = torch.tensor(xt), torch.tensor(yt, dtype=torch.float32)
    vx, vy = torch.tensor(xv), torch.tensor(yv, dtype=torch.float32)
    pos_weight = float((yt == 0).sum() / (yt == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    # Adam + L2 (weight_decay acoplado); não chamar de AdamW.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=protocol["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss, best_epoch, wait, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, protocol["max_epochs"] + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Loss não finita.")
            loss.backward()
            optimizer.step()
        # Mesma loss, mesma ponderação e eval() nas duas curvas: comparação justa.
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(tx), ty).item()
            val_loss = loss_fn(model(vx), vy).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ap": average_precision_score(yt, predict_scores(model, xt)),
                "val_ap": average_precision_score(yv, predict_scores(model, xv)),
            }
        )
        if val_loss < best_loss - protocol["min_delta"]:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= protocol["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return (
        model,
        pd.DataFrame(history),
        {
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "best_val_loss": best_loss,
            "pos_weight": pos_weight,
            "parameters": model.n_parameters,
        },
    )
