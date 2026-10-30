"""Aquisição verificável, esquema semântico e divisão sem perfis compartilhados."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "docs/projects/eda"
DATA_PATH = ROOT / "data/raw/online_shoppers_intention.csv"
SOURCE = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
DOWNLOAD = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
SEED = 42
TARGET = "Revenue"
ENGAGEMENT = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]
RATES = ["BounceRates", "ExitRates", "SpecialDay"]
NUMERIC = ENGAGEMENT + RATES
ALL_NUMERIC = ENGAGEMENT + ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATEGORICAL = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURES = NUMERIC + CATEGORICAL
RAW_FEATURES = ALL_NUMERIC + CATEGORICAL
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def verify(data: bytes) -> None:
    """Nunca aceitar silenciosamente uma versão diferente do conjunto."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != SHA256:
        raise ValueError(f"SHA-256 inesperado: {actual}; esperado: {SHA256}")


def download(path: Path = DATA_PATH) -> Path:
    if path.exists():
        verify(path.read_bytes())
        return path
    request = Request(DOWNLOAD, headers={"User-Agent": "Deeplearnig-EDA/1.0"})
    with urlopen(request, timeout=90) as response:
        archive = response.read()
    with ZipFile(io.BytesIO(archive)) as zipped:
        contents = zipped.read("online_shoppers_intention.csv")
    verify(contents)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    verify(path.read_bytes())
    frame = pd.read_csv(path)
    if set(frame.columns) != set(RAW_FEATURES + [TARGET]):
        raise ValueError("O esquema do CSV mudou.")
    if not frame[TARGET].isin([True, False]).all():
        raise ValueError("Revenue precisa conter apenas True e False.")
    frame[TARGET] = frame[TARGET].astype(int)
    # Códigos numéricos nominais não representam distâncias ou ordens.
    for column in CATEGORICAL:
        frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame


def profile_groups(frame: pd.DataFrame) -> np.ndarray:
    """Mesmo vetor de entrada sempre fica no mesmo grupo, independente do alvo."""
    return pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Regra determinística de integridade; não estima estatísticas nem usa o teste.
    clean = frame.drop_duplicates().copy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_pos, test_pos = next(
        splitter.split(clean[FEATURES], clean[TARGET], profile_groups(clean))
    )
    train, test = clean.iloc[train_pos].copy(), clean.iloc[test_pos].copy()
    if set(profile_groups(train)) & set(profile_groups(test)):
        raise AssertionError("Perfis compartilhados entre treino e teste.")
    return clean, train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    print(f"Dados verificados: {download(args.path)}")
