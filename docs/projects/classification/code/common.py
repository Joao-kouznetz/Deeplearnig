"""Contratos compartilhados e reutilização explícita do pipeline da EDA."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[2]
EDA_CODE = ROOT / "docs/projects/eda/code"
sys.path.insert(0, str(EDA_CODE))

from dataset import (  # noqa: E402
    FEATURES,
    SEED,
    SHA256,
    TARGET,
    download,
    load_data,
    profile_groups,
    split_data,
)
from preprocessing import make_preprocessor  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fingerprint():
    """Mudanças no experimento invalidam uma seleção congelada."""
    sources = [PROJECT / "protocol.json"]
    sources += [
        PROJECT / "code" / f for f in ["common.py", "network.py", "training.py", "experiment.py"]
    ]
    sources += [EDA_CODE / f for f in ["dataset.py", "preprocessing.py"]]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def partitions():
    download()
    _, development, test = split_data(load_data())
    # Não substituir o split externo: compará-lo linha a linha com a EDA entregue.
    published = pd.read_csv(ROOT / "docs/projects/eda/tables/split.csv")
    for label, frame in [("treino", development), ("teste", test)]:
        expected = set(published.loc[published.particao.eq(label), "source_row"])
        if set(frame.index) != expected:
            raise ValueError("Split diferente da EDA; confira as versões e a seed.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    itrain, ival = next(
        splitter.split(development[FEATURES], development[TARGET], profile_groups(development))
    )
    train, val = development.iloc[itrain].copy(), development.iloc[ival].copy()
    groups = [set(profile_groups(f)) for f in (train, val, test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise AssertionError("Perfis compartilhados entre partições.")
    return train, val, test


def transformed(train, val):
    pipeline = make_preprocessor()
    xt = pipeline.fit_transform(train[FEATURES]).astype(np.float32)
    xv = pipeline.transform(val[FEATURES]).astype(np.float32)
    if not np.isfinite(xt).all() or not np.isfinite(xv).all():
        raise ValueError("Entrada não finita após o pipeline.")
    return pipeline, xt, xv


__all__ = [
    "FEATURES",
    "TARGET",
    "SHA256",
    "PROJECT",
    "ROOT",
    "SEED",
    "partitions",
    "transformed",
    "fingerprint",
    "read_json",
    "write_json",
]
