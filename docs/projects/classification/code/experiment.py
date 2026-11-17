"""Experimento em duas fases: seleção na validação e avaliação final bloqueada."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import torch
from common import (
    FEATURES,
    PROJECT,
    SHA256,
    TARGET,
    fingerprint,
    partitions,
    read_json,
    transformed,
    write_json,
)
from network import predict_scores
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from training import fit_network


def utc_now():
    return datetime.now(UTC).isoformat()


def select_threshold(y, scores):
    """Maximiza F2 sem consultar teste. Empates preferem menos alarmes."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    values = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def metrics(y, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "ap": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "precision_1": float(precision_score(y, predicted, zero_division=0)),
        "recall_1": float(recall_score(y, predicted, zero_division=0)),
        "f1_1": float(f1_score(y, predicted, zero_division=0)),
        "f2_1": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def baselines():
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
    }


def folders(output):
    for folder in ["tables", "figures", "checkpoints"]:
        (output / folder).mkdir(parents=True, exist_ok=True)


def tune(output, protocol):
    if (output / "tables/evaluation.json").exists():
        raise ValueError(
            "Teste já avaliado: não ajustar novamente. Reproduza em --output-dir artifacts/reproduction."
        )
    folders(output)
    frozen_hash = fingerprint()
    write_json(
        output / "tables/protocol_before_training.json",
        {
            "registered_at_utc": utc_now(),
            "source_fingerprint": frozen_hash,
            "protocol": protocol,
        },
    )
    train, val, test = partitions()
    pd.concat(
        [
            pd.DataFrame({"source_row": f.index, "partition": label})
            for label, f in [("train", train), ("validation", val), ("test", test)]
        ],
        ignore_index=True,
    ).to_csv(output / "tables/split.csv", index=False)
    preprocessor, xt, xv = transformed(train, val)
    yt, yv = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    baseline_rows = []
    for name, estimator in baselines().items():
        estimator.fit(xt, yt)
        scores = estimator.predict_proba(xv)[:, 1]
        threshold = 0.5 if name == "Dummy" else select_threshold(yv, scores)
        baseline_rows.append({"model": name, **metrics(yv, scores, threshold)})
    baseline_table = pd.DataFrame(baseline_rows).sort_values(
        ["ap", "model"], ascending=[False, True]
    )
    baseline_table.to_csv(output / "tables/baselines_validation.csv", index=False)
    rows, histories = [], []
    for config in protocol["configurations"]:
        model, history, metadata = fit_network(
            xt, yt, xv, yv, config, protocol["search_seed"], protocol
        )
        scores = predict_scores(model, xv)
        threshold = select_threshold(yv, scores)
        rows.append(
            {
                **config,
                "hidden": "-".join(map(str, config["hidden"])),
                **metadata,
                **metrics(yv, scores, threshold),
            }
        )
        histories.append(history.assign(config=config["id"]))
        print(
            f"{config['id']}: AP validação={rows[-1]['ap']:.4f}; melhor época={metadata['best_epoch']}",
            flush=True,
        )
    search = pd.DataFrame(rows).sort_values(["ap", "id"], ascending=[False, True])
    search.insert(0, "rank", range(1, len(search) + 1))
    search.to_csv(output / "tables/search.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        output / "tables/history_search.csv", index=False
    )
    winner = next(c for c in protocol["configurations"] if c["id"] == search.iloc[0]["id"])
    write_json(
        output / "tables/selection.json",
        {
            "selected_at_utc": utc_now(),
            "source_fingerprint": frozen_hash,
            "dataset_sha256": SHA256,
            "config": winner,
            "baseline": baseline_table.iloc[0]["model"],
            "baseline_thresholds": {r["model"]: r["threshold"] for r in baseline_rows},
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "n_features": xt.shape[1],
            "train_positive": int(yt.sum()),
            "validation_positive": int(yv.sum()),
            "train_bmi_median": float(
                preprocessor.named_steps["columns"]
                .named_transformers_["log"]
                .named_steps["impute"]
                .statistics_[1]
            ),
            "feature_names": preprocessor.get_feature_names_out().tolist(),
        },
    )


def permutation(model, preprocessor, val, protocol):
    """Permuta a variável original: one-hots e indicador de ausência permanecem juntos."""
    y = val[TARGET].to_numpy()
    reference = average_precision_score(
        y, predict_scores(model, preprocessor.transform(val[FEATURES]).astype(np.float32))
    )
    rng = np.random.default_rng(42)
    rows = []
    for feature in FEATURES:
        for repeat in range(protocol["permutation_repeats"]):
            shuffled = val[FEATURES].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            score = average_precision_score(
                y, predict_scores(model, preprocessor.transform(shuffled).astype(np.float32))
            )
            rows.append({"feature": feature, "repeat": repeat, "ap_drop": reference - score})
    detail = pd.DataFrame(rows)
    return detail, detail.groupby("feature").ap_drop.agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    ).reset_index()


def subgroup_table(test, scores, threshold):
    frames = {
        "age": pd.cut(
            test.age, [-np.inf, 45, 65, np.inf], right=False, labels=["<45", "45–64", ">=65"]
        ).astype(str),
        "hypertension": test.hypertension,
        "gender": test.gender,
        "bmi_missing": test.bmi.isna().map({True: "missing", False: "observed"}),
        "smoking_status": test.smoking_status,
    }
    rows = []
    for feature, categories in frames.items():
        for category in sorted(categories.unique()):
            mask = categories.eq(category).to_numpy()
            y, p = test[TARGET].to_numpy()[mask], scores[mask] >= threshold
            positives = int(y.sum())
            rows.append(
                {
                    "feature": feature,
                    "group": category,
                    "n": int(mask.sum()),
                    "positive": positives,
                    "error_rate": float((y != p).mean()),
                    "fn": int(((y == 1) & ~p).sum()),
                    "fp": int(((y == 0) & p).sum()),
                    "recall_1": float(recall_score(y, p, zero_division=0)) if positives else None,
                    "f1_1": float(f1_score(y, p, zero_division=0)) if positives else None,
                }
            )
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)


def evaluate(output, protocol):
    if (output / "tables/evaluation.json").exists():
        raise ValueError("Avaliação final já existe; use report ou outra pasta para reprodução.")
    selected = read_json(output / "tables/selection.json")
    if selected["source_fingerprint"] != fingerprint():
        raise ValueError(
            "Código/protocolo mudou desde a seleção. Não reutilizar o teste para ajustar."
        )
    train, val, test = partitions()
    preprocessor, xt, xv = transformed(train, val)
    yt, yv = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    networks, final_meta, histories = {}, [], []
    # Todos os treinos e limiares terminam antes de calcular qualquer métrica de teste.
    for seed in protocol["final_seeds"]:
        model, history, meta = fit_network(xt, yt, xv, yv, selected["config"], seed, protocol)
        threshold = select_threshold(yv, predict_scores(model, xv))
        networks[seed] = model
        final_meta.append({"seed": seed, "threshold": threshold, **meta})
        histories.append(history.assign(seed=seed))
        torch.save(model.state_dict(), output / f"checkpoints/seed_{seed}.pt")
    estimators = baselines()
    for estimator in estimators.values():
        estimator.fit(xt, yt)
    joblib.dump(preprocessor, output / "checkpoints/preprocessor.joblib")
    write_json(
        output / "tables/final_before_test.json",
        {
            "frozen_at_utc": utc_now(),
            "source_fingerprint": fingerprint(),
            "training": final_meta,
            "baseline_thresholds": selected["baseline_thresholds"],
        },
    )
    detail, importance = permutation(networks[42], preprocessor, val, protocol)
    detail.to_csv(output / "tables/permutation_repeats.csv", index=False)
    importance.to_csv(output / "tables/permutation.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "tables/history_final.csv", index=False)
    # Abertura final única, em lote: mesmas 1.022 linhas, sem seleção entre seeds.
    x_test = preprocessor.transform(test[FEATURES]).astype(np.float32)
    y_test = test[TARGET].to_numpy()
    rows, predictions, class_rows = [], [], []
    models = [
        (name, 42, estimator.predict_proba(x_test)[:, 1], selected["baseline_thresholds"][name])
        for name, estimator in estimators.items()
    ]
    models += [
        ("MLP", meta["seed"], predict_scores(networks[meta["seed"]], x_test), meta["threshold"])
        for meta in final_meta
    ]
    for name, seed, scores, threshold in models:
        rows.append({"model": name, "seed": seed, **metrics(y_test, scores, threshold)})
        predictions.append(
            pd.DataFrame(
                {
                    "source_row": test.index,
                    "model": name,
                    "seed": seed,
                    "y": y_test,
                    "score": scores,
                    "predicted": (scores >= threshold).astype(int),
                }
            )
        )
        by_class = classification_report(
            y_test, scores >= threshold, labels=[0, 1], output_dict=True, zero_division=0
        )
        for label in ["0", "1"]:
            class_rows.append({"model": name, "seed": seed, "class": label, **by_class[label]})
    result = pd.DataFrame(rows)
    result.to_csv(output / "tables/test_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "tables/test_per_class.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / "tables/test_predictions.csv", index=False)
    main_prediction = predictions.query("model == 'MLP' and seed == 42")
    subgroup_table(test, main_prediction.score.to_numpy(), final_meta[0]["threshold"]).to_csv(
        output / "tables/subgroups.csv", index=False
    )
    errors = main_prediction[main_prediction.y.ne(main_prediction.predicted)].copy()
    # Confiança no erro = distância do limiar no lado incorreto, não probabilidade calibrada.
    errors["distance_to_threshold"] = (errors.score - final_meta[0]["threshold"]).abs()
    worst = pd.concat(
        [
            errors.query("y == 1").sort_values("distance_to_threshold", ascending=False).head(5),
            errors.query("y == 0").sort_values("distance_to_threshold", ascending=False).head(5),
        ]
    )
    # Publicar só descritores necessários à discussão, sem id ou reprodução do CSV bruto.
    worst = worst.merge(
        test[["age", "avg_glucose_level", "bmi", "hypertension"]],
        left_on="source_row",
        right_index=True,
    )
    worst.to_csv(output / "tables/worst_errors.csv", index=False)
    mlp = result.query("model == 'MLP'")
    baseline_ap = float(result.loc[result.model.eq(selected["baseline"]), "ap"].iloc[0])
    mean, std = float(mlp.ap.mean()), float(mlp.ap.std(ddof=1))
    write_json(
        output / "tables/evaluation.json",
        {
            "evaluated_at_utc": utc_now(),
            "source_fingerprint": fingerprint(),
            "baseline": selected["baseline"],
            "baseline_ap": baseline_ap,
            "mlp_ap_mean": mean,
            "mlp_ap_std": std,
            "gap": mean - baseline_ap,
            "relative_gain": mean / baseline_ap - 1,
            "success": bool(mean >= (1 + protocol["success_relative_improvement"]) * baseline_ap),
            "test_rows": len(test),
            "test_positive": int(y_test.sum()),
            "final_training": final_meta,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scikit-learn", "torch", "scipy", "matplotlib"]
            },
            "test_policy": "Uma avaliação em lote de modelos e seeds pré-fixados; nenhum ajuste posterior ao teste.",
        },
    )
    print(
        f"Teste: AP MLP={mean:.4f} ± {std:.4f}; {selected['baseline']}={baseline_ap:.4f}; sucesso={mean >= 1.1 * baseline_ap}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "evaluate", "report", "all"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    folders(output)
    protocol = read_json(PROJECT / "protocol.json")
    if args.stage in ["tune", "all"]:
        tune(output, protocol)
    if args.stage in ["evaluate", "all"]:
        evaluate(output, protocol)
    if args.stage in ["report", "all"]:
        from stroke_report import build_report

        build_report(output, protocol)


if __name__ == "__main__":
    main()
