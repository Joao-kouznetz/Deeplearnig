"""Validação dos dados reais, rastreabilidade e vínculos do relatório publicado."""

import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from dataset import (
    FEATURES,
    PROJECT,
    ROOT,
    SHA256,
    TARGET,
    download,
    load_data,
    profile_groups,
    split_data,
    verify,
)
from preprocessing import make_preprocessor


def test_dataset_integrity_rejects_changed_content(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        verify(b"conteudo diferente")
    corrupt = tmp_path / "online_shoppers_intention.csv"
    corrupt.write_bytes(b"arquivo corrompido")
    with pytest.raises(ValueError, match="SHA-256"):
        download(corrupt)
    assert corrupt.read_bytes() == b"arquivo corrompido"


def test_real_data_partition_and_pipeline_match_reported_metrics():
    raw = load_data()
    clean, train, test = split_data(raw)
    assert raw.shape == (12330, 18)
    assert len(clean) == 12205
    assert raw[TARGET].sum() == 1908
    assert not set(profile_groups(train)) & set(profile_groups(test))
    pipeline = make_preprocessor()
    train_t = pipeline.fit_transform(train[FEATURES])
    test_t = pipeline.transform(test[FEATURES])
    metrics = json.loads((PROJECT / "tables/metrics.json").read_text())
    assert metrics["dataset_sha256"] == SHA256
    assert metrics["train_rows"] == len(train)
    assert metrics["test_rows"] == len(test)
    assert metrics["train_shape"] == list(train_t.shape)
    assert metrics["test_shape"] == list(test_t.shape)
    assert np.isfinite(train_t).all() and np.isfinite(test_t).all()
    names = pd.read_csv(PROJECT / "tables/feature_names.csv")["feature"].tolist()
    assert names == pipeline.get_feature_names_out().tolist()
    membership = pd.read_csv(PROJECT / "tables/split.csv")
    assert set(membership.query("particao == 'treino'").source_row) == set(train.index)
    assert set(membership.query("particao == 'teste'").source_row) == set(test.index)


def test_report_front_matter_figures_code_and_results_summary():
    text = (PROJECT / "index.md").read_text()
    front_matter = yaml.safe_load(text.split("---", 2)[1])
    assert front_matter["project"] == "eda"
    assert front_matter["task"] == "classification"
    assert "inteligência artificial" in front_matter["ai_use"]
    headings = re.findall(r"^## ([0-5])\.", text, re.MULTILINE)
    assert headings == list("012345")
    figures = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", text)
    assert len(figures) == 9
    for figure in figures:
        assert (PROJECT / figure).is_file()
    for link in re.findall(r"\]\((tables/[^)]+)\)", text):
        assert (PROJECT / link).is_file()
    summary = pd.read_csv(PROJECT / "tables/results_summary.csv")
    assert summary["#"].tolist() == list(range(1, 11))
    assert not summary.isna().any().any()
    for required in ["dataset.py", "preprocessing.py", "analysis.py", "reporting.py", "analyze.py"]:
        assert (PROJECT / "code" / required).is_file()
    assert (ROOT / "README.md").is_file()


def test_projection_sample_is_training_only_and_all_five_maps_use_it():
    coordinates = pd.read_csv(PROJECT / "tables/projections.csv")
    membership = pd.read_csv(PROJECT / "tables/split.csv")
    train_indices = set(membership.query("particao == 'treino'").source_row)
    assert len(coordinates) == 2000
    assert coordinates.source_row.is_unique
    assert set(coordinates.source_row).issubset(train_indices)
    assert np.isfinite(coordinates.drop(columns=["source_row", "Revenue"]).to_numpy()).all()
    for projection in ["PCA", "t-SNE p=15", "t-SNE p=50", "UMAP k=15", "UMAP k=50"]:
        assert all(f"{projection}_{axis}" in coordinates for axis in [1, 2])
    diagnostics = pd.read_csv(PROJECT / "tables/projection_diagnostics.csv")
    assert diagnostics.trustworthiness.between(0, 1).all()
    assert diagnostics.n.eq(2000).all()


def test_built_site_serves_the_figures_and_downloads_linked_in_the_report(tmp_path):
    destination = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(destination)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    markdown = (PROJECT / "index.md").read_text()
    assets = re.findall(r"\]\(((?:tables|figures)/[^)]+)\)", markdown)
    assert len(assets) >= 13
    for asset in assets:
        published = destination / "projects/eda" / asset
        assert published.is_file(), f"Link publicado sem arquivo: {asset}"
        assert published.read_bytes() == (PROJECT / asset).read_bytes()
    assert not list(destination.rglob("*.pyc"))
