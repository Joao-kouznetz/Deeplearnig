"""Contratos do experimento: isolamento, checkpoints e números publicados."""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from common import PROJECT, ROOT, TARGET, fingerprint, partitions, transformed
from experiment import evaluate, metrics, select_threshold, tune
from network import predict_scores
from sklearn.metrics import fbeta_score
from training import fit_network


def read(name):
    return json.loads((PROJECT / "tables" / name).read_text())


def test_internal_split_preserves_external_eda_and_fits_only_internal_train():
    train, val, test = partitions()
    assert (len(train), len(val), len(test)) == (3270, 818, 1022)
    assert [int(f[TARGET].sum()) for f in (train, val, test)] == [159, 40, 50]
    published = pd.read_csv(ROOT / "docs/projects/eda/tables/split.csv")
    assert set(test.index) == set(published.query("particao == 'teste'").source_row)
    assert set(train.index) | set(val.index) == set(
        published.query("particao == 'treino'").source_row
    )
    poisoned_val = val.copy()
    poisoned_val["age"] = 1e6
    poisoned_val["bmi"] = 1e5
    poisoned_val["gender"] = "unseen_validation_only"
    preprocessor, xt, xv = transformed(train, poisoned_val)
    blocks = preprocessor.named_steps["columns"].named_transformers_
    assert blocks["direct"].named_steps["scale"].mean_[0] == pytest.approx(train.age.mean())
    assert blocks["log"].named_steps["impute"].statistics_[1] == pytest.approx(train.bmi.median())
    assert "unseen_validation_only" not in blocks["category"].named_steps["onehot"].categories_[0]
    assert xt.shape[1] == xv.shape[1] == 24
    assert np.isfinite(xt).all() and np.isfinite(xv).all()


def test_training_restores_best_loss_and_switches_off_dropout():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    config = {"hidden": [8], "dropout": 0.2, "weight_decay": 0.0, "lr": 0.03}
    protocol = {"batch_size": 16, "max_epochs": 40, "patience": 3, "min_delta": 0.0}
    model, history, meta = fit_network(x, y, x, 1 - y, config, 42, protocol)
    assert meta["stopped_epoch"] < protocol["max_epochs"]
    assert meta["best_epoch"] < meta["stopped_epoch"]
    assert not model.training
    with torch.no_grad():
        actual_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(meta["pos_weight"]))(
            model(torch.tensor(x)), torch.tensor(1 - y, dtype=torch.float32)
        ).item()
    assert actual_loss == pytest.approx(meta["best_val_loss"], abs=1e-6)
    assert actual_loss == pytest.approx(history.val_loss.min(), abs=1e-6)
    np.testing.assert_array_equal(predict_scores(model, x), predict_scores(model, x))


def test_threshold_is_optimal_f2_on_its_input_including_ties():
    y = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.1, 0.3, 0.3, 0.8, 0.7, 0.9])
    threshold = select_threshold(y, scores)
    achieved = fbeta_score(y, scores >= threshold, beta=2)
    assert achieved == max(fbeta_score(y, scores >= t, beta=2) for t in np.unique(scores))


def test_published_metrics_are_recomputed_from_predictions():
    predictions = pd.read_csv(PROJECT / "tables/test_predictions.csv")
    reported = pd.read_csv(PROJECT / "tables/test_metrics.csv")
    reference_rows = None
    for r in reported.itertuples():
        part = predictions[predictions.model.eq(r.model) & predictions.seed.eq(r.seed)]
        assert len(part) == 1022 and part.source_row.is_unique
        assert part.y.sum() == 50
        if reference_rows is None:
            reference_rows = set(part.source_row)
        assert set(part.source_row) == reference_rows
        computed = metrics(part.y.to_numpy(), part.score.to_numpy(), r.threshold)
        for key, value in computed.items():
            assert getattr(r, key) == pytest.approx(value, abs=1e-6)
        np.testing.assert_array_equal(part.predicted, (part.score >= r.threshold).astype(int))
    summary = read("evaluation.json")
    mlp = reported[reported.model.eq("MLP")]
    assert mlp.seed.tolist() == [42, 43, 44]
    assert summary["mlp_ap_mean"] == pytest.approx(mlp.ap.mean())
    assert summary["mlp_ap_std"] == pytest.approx(mlp.ap.std(ddof=1))
    assert summary["success"] == bool(mlp.ap.mean() >= 1.1 * summary["baseline_ap"])


def test_search_and_frozen_protocol_precede_test_and_match_code():
    protocol = read("protocol_before_training.json")
    selection = read("selection.json")
    final = read("final_before_test.json")
    evaluation = read("evaluation.json")
    times = [
        datetime.fromisoformat(d[k])
        for d, k in [
            (protocol, "registered_at_utc"),
            (selection, "selected_at_utc"),
            (final, "frozen_at_utc"),
            (evaluation, "evaluated_at_utc"),
        ]
    ]
    assert times == sorted(times)
    assert {d["source_fingerprint"] for d in [protocol, selection, final, evaluation]} == {
        fingerprint()
    }
    search = pd.read_csv(PROJECT / "tables/search.csv")
    assert len(search) >= 8
    assert search.ap.is_monotonic_decreasing
    assert selection["config"]["id"] == search.iloc[0].id
    baseline = pd.read_csv(PROJECT / "tables/baselines_validation.csv")
    assert selection["baseline"] == baseline.sort_values("ap", ascending=False).iloc[0].model
    ablation = search.set_index("id").loc[["c01", "c02", "c03", "c04"]]
    assert ablation.hidden.nunique() == ablation.lr.nunique() == 1
    assert ablation.dropout.nunique() >= 2 and ablation.weight_decay.nunique() >= 2


def test_test_guard_refuses_retuning_or_second_evaluation():
    with pytest.raises(ValueError, match="Teste já avaliado"):
        tune(PROJECT, {})
    with pytest.raises(ValueError, match="já existe"):
        evaluate(PROJECT, {})


def test_subgroups_and_worst_examples_match_published_predictions():
    _, _, test = partitions()
    predictions = (
        pd.read_csv(PROJECT / "tables/test_predictions.csv")
        .query("model == 'MLP' and seed == 42")
        .set_index("source_row")
    )
    assert set(predictions.index) == set(test.index)
    subgroups = pd.read_csv(PROJECT / "tables/subgroups.csv")
    groups = subgroups[subgroups.feature.eq("age")].set_index("group")
    ages = pd.cut(test.age, [-np.inf, 45, 65, np.inf], right=False, labels=["<45", "45–64", ">=65"])
    for group, observed in groups.iterrows():
        p = predictions.loc[test.index[ages.eq(group)]]
        assert observed.n == len(p)
        assert observed.error_rate == pytest.approx(p.y.ne(p.predicted).mean())
    worst = pd.read_csv(PROJECT / "tables/worst_errors.csv")
    assert len(worst) == 10
    for r in worst.itertuples():
        p = predictions.loc[r.source_row]
        assert p.y != p.predicted
        assert r.score == pytest.approx(p.score)


def test_classification_report_has_all_steps_assets_and_filled_summary():
    report = (PROJECT / "index.md").read_text()
    front = yaml.safe_load(report.split("---", 2)[1])
    assert front["project"] == "classification"
    assert front["team"] and "Codex" in front["ai_use"]
    assert re.findall(r"^## ([0-7])\.", report, re.MULTILINE) == list("01234567")
    assert report.rsplit("\n## ", 1)[1].startswith("Results summary")
    assets = re.findall(r"\]\(((?:figures|tables|code)/[^)]+)\)", report)
    assert len([a for a in assets if a.startswith("figures/")]) == 7
    for asset in assets:
        assert (PROJECT / asset).is_file()
    summary = pd.read_csv(PROJECT / "tables/results_summary.csv")
    assert summary["#"].tolist() == list(range(1, 9))
    assert not summary.isna().any().any()


def test_reproduction_matches_reference_when_available():
    path = ROOT / "artifacts/reproduction/tables/test_metrics.csv"
    if not path.exists():
        pytest.skip("Reprodução completa não executada nesta máquina.")
    reference = pd.read_csv(PROJECT / "tables/test_metrics.csv").sort_values(["model", "seed"])
    actual = pd.read_csv(path).sort_values(["model", "seed"])
    assert reference[["model", "seed"]].values.tolist() == actual[["model", "seed"]].values.tolist()
    # Tolerância para diferenças numéricas de CPU/BLAS; não reescolher a rede no teste.
    np.testing.assert_allclose(reference.ap, actual.ap, atol=0.002, rtol=0)
