"""Reproduz tabelas, figuras, relatório e pipeline, sem treinar um modelo preditivo."""

from __future__ import annotations

import json
import os

# Antes de importar as bibliotecas científicas: execução determinística e modesta em memória.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from analysis import descriptive_tables, eda_figures, preprocessing_diagnostics, projections
from dataset import (
    CATEGORICAL,
    FEATURES,
    PROJECT,
    ROOT,
    SHA256,
    TARGET,
    download,
    load_data,
    split_data,
)
from preprocessing import make_preprocessor
from reporting import report


def pipeline_checks(preprocessor, train, test, train_matrix, test_matrix):
    if not np.isfinite(train_matrix).all() or not np.isfinite(test_matrix).all():
        raise AssertionError("A saída contém NaN ou infinito.")
    names = preprocessor.get_feature_names_out()
    if any(name.endswith("__id") or name.endswith(f"__{TARGET}") for name in names):
        raise AssertionError("Coluna descartada vazou para a saída.")
    probe = test[FEATURES].iloc[:2].copy()
    probe["work_type"] = "Nunca_observado"
    probe.loc[probe.index[0], "bmi"] = np.nan
    probe.loc[probe.index[0], "gender"] = np.nan
    transformed = preprocessor.transform(probe)
    unknown_columns = np.array([name.startswith("category__work_type_") for name in names])
    if (
        transformed.shape[1] != len(names)
        or not np.isfinite(transformed).all()
        or transformed[:, unknown_columns].any()
    ):
        raise AssertionError("Falhou o contrato para ausentes/categoria inédita.")
    unseen = {col: sorted(set(test[col]) - set(train[col])) for col in CATEGORICAL}
    unseen = {col: values for col, values in unseen.items() if values}
    return {
        "unseen_test_categories": json.dumps(unseen, ensure_ascii=False) if unseen else "Nenhuma",
        "finite": True,
        "synthetic_unknown_passed": True,
        "synthetic_missing_passed": True,
    }


def main():
    for folder in [PROJECT / "tables", PROJECT / "figures", ROOT / "artifacts"]:
        folder.mkdir(parents=True, exist_ok=True)
    download()
    raw = load_data()
    clean, train, test = split_data(raw)
    print(f"Registros: bruto={len(raw)}, treino={len(train)}, teste={len(test)}", flush=True)
    tables = descriptive_tables(raw, clean, train, test)
    preprocessor = make_preprocessor()
    train_matrix = preprocessor.fit_transform(train[FEATURES])
    test_matrix = preprocessor.transform(test[FEATURES])
    names = preprocessor.get_feature_names_out()
    checks = pipeline_checks(preprocessor, train, test, train_matrix, test_matrix)
    pd.DataFrame({"feature": names}).to_csv(PROJECT / "tables/feature_names.csv", index=False)
    print(f"Pipeline: {train_matrix.shape} / {test_matrix.shape}; saídas finitas", flush=True)
    tables["scaling_checks"] = preprocessing_diagnostics(train, train_matrix, test_matrix, names)
    eda_figures(raw, train, tables)
    pca, diagnostics, positions = projections(train_matrix, train)
    report(
        raw, clean, train, test, tables, pca, diagnostics, names, train_matrix, test_matrix, checks
    )
    metrics = tables["summary_counts"] | {
        "seed": 42,
        "dataset_sha256": SHA256,
        "train_shape": list(train_matrix.shape),
        "test_shape": list(test_matrix.shape),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "projection_sample_rows": len(positions),
        "checks": checks,
        "projections": diagnostics.to_dict(orient="records"),
    }
    (PROJECT / "tables/metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Binários regeneráveis ficam fora do Git; o código de reconstrução é versionado.
    joblib.dump(preprocessor, ROOT / "artifacts/preprocessor.joblib")
    print("Relatório gerado: docs/projects/eda/index.md", flush=True)


if __name__ == "__main__":
    main()
