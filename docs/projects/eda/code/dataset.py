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
FILENAME = "healthcare-dataset-stroke-data.csv"
DATA_PATH = ROOT / "data/raw" / FILENAME
SOURCE = "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/stroke-prediction-dataset?dataset_version_number=1"
SHA256 = "644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420"
SEED = 42
TARGET = "stroke"
LOG_FEATURES = ["avg_glucose_level", "bmi"]
DIRECT_NUMERIC = ["age"]
NUMERIC = DIRECT_NUMERIC + LOG_FEATURES
CATEGORICAL = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
FEATURES = NUMERIC + CATEGORICAL
RAW_FEATURES = ["id"] + FEATURES


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
        contents = zipped.read(FILENAME)
    verify(contents)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    verify(path.read_bytes())
    frame = pd.read_csv(path)
    if set(frame.columns) != set(RAW_FEATURES + [TARGET]):
        raise ValueError("O esquema do CSV mudou.")
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("stroke precisa conter apenas 0 e 1.")
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
    clean = frame.drop_duplicates(subset=FEATURES + [TARGET]).copy()
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
