"""Estatísticas e figuras da EDA de AVC, sem usar o teste nas etapas 2–4."""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import CATEGORICAL, LOG_FEATURES, NUMERIC, PROJECT, SEED, TARGET, profile_groups
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from umap import UMAP

COLORS = ["#416788", "#C65D21"]
CLASS_NAMES = ["Sem AVC registrado", "Com AVC registrado"]
LABELS = [
    Line2D([0], [0], marker="o", linestyle="", color=c, label=n)
    for c, n in zip(COLORS, CLASS_NAMES, strict=True)
]
AXIS_LABELS = {
    "age": "Idade (anos)",
    "bmi": "IMC (kg/m²)",
    "avg_glucose_level": "Glicose média (unidade não informada)",
}


def save_figure(fig, name, output=PROJECT):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output / "figures" / f"{name}.png",
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "Deeplearnig EDA"},
    )
    plt.close(fig)


def ordered_values(frame, column):
    return sorted(frame[column].dropna().unique(), key=lambda x: (len(x), x))


def wilson(success, total):
    z = 1.96
    proportion = success / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return center - half, center + half


def descriptive_tables(raw, clean, train, test, output=PROJECT):
    tables = {}
    # Unknown é uma categoria explícita, mas a fonte a define como informação ausente.
    missing = raw.isna()
    declared_missing = missing.copy()
    declared_missing["smoking_status"] |= raw.smoking_status.eq("Unknown")
    tables["quality"] = pd.DataFrame(
        {
            "ausentes_n": missing.sum(),
            "ausentes_pct": missing.mean() * 100,
            "ausencia_declarada_n": declared_missing.sum(),
            "ausencia_declarada_pct": declared_missing.mean() * 100,
            "cardinalidade": raw.nunique(),
        }
    )
    tables["quality"].index.name = "variavel"
    numeric = train[NUMERIC].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
    numeric["skewness"] = train[NUMERIC].skew()
    numeric["missing_n"] = train[NUMERIC].isna().sum()
    tables["numeric"] = numeric
    tables["categorical"] = pd.DataFrame(
        [
            {
                "variavel": col,
                "categoria": value,
                "n": int(count),
                "pct": count / len(train) * 100,
                "rara_menos_1pct": bool(count / len(train) < 0.01),
            }
            for col in CATEGORICAL
            for value, count in train[col].value_counts().items()
        ]
    )
    tables["cardinality"] = pd.DataFrame(
        [
            {
                "variavel": col,
                "cardinalidade": train[col].nunique(),
                "categorias_raras": int((train[col].value_counts(normalize=True) < 0.01).sum()),
            }
            for col in CATEGORICAL
        ]
    )
    corr = train[NUMERIC].corr(method="spearman")
    tables["correlation"] = corr
    tables["correlation_pairs"] = pd.DataFrame(
        [
            {
                "variavel_1": a,
                "variavel_2": b,
                "rho": float(corr.loc[a, b]),
                "n_pares_completos": int(train[[a, b]].dropna().shape[0]),
            }
            for i, a in enumerate(NUMERIC)
            for b in NUMERIC[i + 1 :]
        ]
    ).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    conditional = []
    for col in CATEGORICAL:
        for cat in ordered_values(train, col):
            values = train.loc[train[col].eq(cat), TARGET]
            low, high = wilson(values.sum(), len(values))
            conditional.append(
                {
                    "variavel": col,
                    "categoria": cat,
                    "n": len(values),
                    "avc_n": int(values.sum()),
                    "taxa_pct": values.mean() * 100,
                    "wilson_low_pct": low * 100,
                    "wilson_high_pct": high * 100,
                }
            )
    tables["category_target"] = pd.DataFrame(conditional)
    box_rows = []
    for group_value, group in train.groupby("hypertension"):
        for col in ["age", "avg_glucose_level"]:
            box_rows.append(
                {
                    "hypertension": group_value,
                    "variavel": col,
                    "n": len(group),
                    "q1": group[col].quantile(0.25),
                    "mediana": group[col].median(),
                    "q3": group[col].quantile(0.75),
                    "IQR": group[col].quantile(0.75) - group[col].quantile(0.25),
                }
            )
    tables["grouped_numeric"] = pd.DataFrame(box_rows)
    q1, q3 = train[NUMERIC].quantile(0.25), train[NUMERIC].quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flags = train[NUMERIC].lt(lower) | train[NUMERIC].gt(upper)
    tables["outliers"] = pd.DataFrame(
        {
            "q1": q1,
            "q3": q3,
            "limiar_inferior": lower,
            "limiar_superior": upper,
            "fora_IQR_n": flags.sum(),
        }
    )
    invalid = [
        {"regra": "id repetido", "n": int(raw.id.duplicated().sum())},
        {"regra": "age < 0 ou > 120", "n": int((raw.age.lt(0) | raw.age.gt(120)).sum())},
        {"regra": "avg_glucose_level <= 0", "n": int(raw.avg_glucose_level.le(0).sum())},
        {"regra": "bmi <= 0 (entre observados)", "n": int(raw.bmi.le(0).sum())},
    ]
    for col in ["hypertension", "heart_disease"]:
        invalid.append(
            {"regra": f"{col} fora de 0/1", "n": int((~raw[col].isin(["0", "1"])).sum())}
        )
    tables["domain_checks"] = pd.DataFrame(invalid)
    tables["classes"] = pd.DataFrame(
        [
            {
                "particao": name,
                "classe": cls,
                "n": int(frame[TARGET].eq(cls).sum()),
                "pct": frame[TARGET].eq(cls).mean() * 100,
            }
            for name, frame in [
                ("bruto", raw),
                ("sem_duplicatas", clean),
                ("treino", train),
                ("teste", test),
            ]
            for cls in [0, 1]
        ]
    )
    tables["missingness_target"] = pd.DataFrame(
        [
            {
                "stroke": cls,
                "n": len(group),
                "bmi_ausente_n": int(group.bmi.isna().sum()),
                "bmi_ausente_pct": group.bmi.isna().mean() * 100,
                "tabagismo_unknown_n": int(group.smoking_status.eq("Unknown").sum()),
                "tabagismo_unknown_pct": group.smoking_status.eq("Unknown").mean() * 100,
            }
            for cls, group in train.groupby(TARGET)
        ]
    )
    tables["split"] = pd.concat(
        [
            pd.DataFrame(
                {
                    "source_row": part.index,
                    "particao": name,
                    "profile_group": profile_groups(part).astype(str),
                }
            )
            for name, part in [("treino", train), ("teste", test)]
        ]
    ).sort_values("source_row")
    tables["summary_counts"] = {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "duplicates_removed": len(raw) - len(clean),
        "train_rows": len(train),
        "test_rows": len(test),
        "profile_groups": int(len(np.unique(profile_groups(clean)))),
        "repeated_profile_rows": int(len(clean) - len(np.unique(profile_groups(clean)))),
        "outlier_train_rows": int(flags.any(axis=1).sum()),
        "log_changed_train_rows": int(train[LOG_FEATURES].gt(0).any(axis=1).sum()),
        "missing_total": int(missing.sum().sum()),
        "unknown_smoking_rows": int(raw.smoking_status.eq("Unknown").sum()),
        "train_bmi_missing": int(train.bmi.isna().sum()),
        "train_bmi_median": float(train.bmi.median()),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(
                output / "tables" / f"{name}.csv",
                index=not isinstance(table.index, pd.RangeIndex),
                float_format="%.10g",
            )
    return tables


def eda_figures(raw, train, tables, output=PROJECT):
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = raw[TARGET].value_counts().reindex([0, 1])
    bars = ax.bar(CLASS_NAMES, counts, color=COLORS, label=CLASS_NAMES)
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(raw):.2%})" for n in counts], padding=4)
    ax.set(xlabel="AVC registrado na base", ylabel="Registros", ylim=(0, counts.max() * 1.2))
    ax.set_title("Figura 1 — Distribuição de stroke no arquivo original")
    ax.legend()
    save_figure(fig, "01_target", output)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, NUMERIC, strict=True):
        values = train[col].dropna()
        ax.hist(values, bins=30, color=COLORS[0], label=f"Treino observado (n={len(values)})")
        ax.set(
            title=f"{col} | assimetria={values.skew():.2f}",
            xlabel=AXIS_LABELS[col],
            ylabel="Registros",
        )
        ax.legend(fontsize=7)
    fig.suptitle("Figura 2 — Distribuições numéricas antes de imputação e log1p (treino)")
    save_figure(fig, "02_numeric", output)

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for ax, col in zip(axes.flat, CATEGORICAL, strict=False):
        freq = train[col].value_counts(normalize=True).reindex(ordered_values(train, col)) * 100
        ax.bar(freq.index, freq, color=COLORS[0], label="Treino")
        ax.set(title=col, xlabel="Categoria", ylabel="Registros (%)")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
    axes.flat[-1].set_visible(False)
    fig.suptitle("Figura 3 — Frequências dos sete atributos categóricos (treino)")
    save_figure(fig, "03_categorical", output)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        tables["correlation"],
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        ax=ax,
        cbar_kws={"label": "Spearman (ρ)"},
    )
    ax.set(
        xlabel="Variável numérica",
        ylabel="Variável numérica",
        title="Figura 4 — Correlações por pares observados (treino)",
    )
    save_figure(fig, "04_correlation", output)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (a, b) in zip(axes, [("age", "bmi"), ("age", "avg_glucose_level")], strict=True):
        for cls, color, label in zip([0, 1], COLORS, CLASS_NAMES, strict=True):
            group = train[train[TARGET].eq(cls)]
            ax.scatter(group[a], group[b], s=9, alpha=0.35, c=color, label=label)
        ax.set(
            xlabel=AXIS_LABELS[a],
            ylabel=AXIS_LABELS[b],
            title=f"ρ={tables['correlation'].loc[a, b]:.3f}",
        )
        ax.legend(fontsize=8)
    fig.suptitle("Figura 5 — Idade, IMC e glicose, por alvo (treino; sem imputação)")
    save_figure(fig, "05_scatter", output)

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for ax, col in zip(axes.flat, CATEGORICAL, strict=False):
        tab = tables["category_target"].query("variavel == @col")
        x = np.arange(len(tab))
        ax.vlines(
            x, tab.wilson_low_pct, tab.wilson_high_pct, color=COLORS[0], label="IC Wilson 95%"
        )
        ax.scatter(x, tab.taxa_pct, color=COLORS[1], label="AVC observado", zorder=3)
        ax.axhline(
            train[TARGET].mean() * 100, color="gray", linestyle="--", label="Média do treino"
        )
        ax.set(
            xticks=x,
            xticklabels=tab.categoria,
            xlabel="Categoria",
            ylabel="AVC registrado (%)",
            title=col,
            ylim=(0, max(20, tab.wilson_high_pct.max() + 3)),
        )
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Figura 6 — Frequência do alvo por categoria, com IC (treino)")
    save_figure(fig, "06_category_target", output)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col in zip(axes, ["age", "avg_glucose_level"], strict=True):
        sns.boxplot(
            data=train,
            x="hypertension",
            y=col,
            hue=TARGET,
            palette=COLORS,
            order=ordered_values(train, "hypertension"),
            showfliers=False,
            ax=ax,
        )
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, CLASS_NAMES, title="stroke", fontsize=8)
        ax.set(xlabel="Hipertensão (0 = não; 1 = sim)", ylabel=AXIS_LABELS[col])
    fig.suptitle("Figura 7 — Localização e dispersão por hipertensão e alvo (treino)")
    save_figure(fig, "07_grouped", output)


def projections(matrix, train, output=PROJECT, sample_size=2000):
    """Mesmo subconjunto para os cinco mapas; y só colore, nunca ajusta projeções."""
    pca = PCA(svd_solver="full", random_state=SEED).fit(matrix)
    positions = np.arange(len(train))
    if len(train) > sample_size:
        positions, _ = train_test_split(
            positions, train_size=sample_size, stratify=train[TARGET], random_state=SEED
        )
    sample = matrix[positions]
    labels = train.iloc[positions][TARGET].to_numpy()
    embeddings = {"PCA": pca.transform(sample)[:, :2]}
    for perplexity in [15, 50]:
        print(f"t-SNE: perplexity={perplexity}", flush=True)
        embeddings[f"t-SNE p={perplexity}"] = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=SEED,
            n_jobs=1,
        ).fit_transform(sample)
    for neighbors in [15, 50]:
        print(f"UMAP: n_neighbors={neighbors}", flush=True)
        embeddings[f"UMAP k={neighbors}"] = UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=SEED,
            transform_seed=SEED,
            n_jobs=1,
        ).fit_transform(sample)

    fig, ax = plt.subplots(figsize=(8, 4))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative * 100,
        marker=".",
        label="PCA (treino completo)",
    )
    ax.axvline(2, color=COLORS[1], linestyle="--", label=f"PC1+PC2: {cumulative[1]:.2%}")
    ax.axhline(90, color="gray", linestyle=":", label="90% de variância")
    ax.set(
        xlabel="Número de componentes",
        ylabel="Variância acumulada (%)",
        ylim=(0, 102),
        title="Figura 8 — Variância explicada da matriz preparada",
    )
    ax.legend()
    save_figure(fig, "08_pca_variance", output)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, embedding) in zip(axes.flat, embeddings.items(), strict=False):
        for cls, color in zip([0, 1], COLORS, strict=True):
            points = embedding[labels == cls]
            ax.scatter(points[:, 0], points[:, 1], s=7, alpha=0.45, c=color, rasterized=True)
        ax.set(
            xlabel="Componente 1" if name == "PCA" else "Coordenada 1",
            ylabel="Componente 2" if name == "PCA" else "Coordenada 2",
            title=name,
        )
        ax.legend(handles=LABELS, fontsize=8)
    axes.flat[-1].set_visible(False)
    fig.suptitle(
        f"Figura 9 — Mesmas {len(sample):,} registros de treino em cinco projeções 2D (seed=42)"
    )
    save_figure(fig, "09_projections", output)

    diagnostics = []
    for name, embedding in embeddings.items():
        diagnostics.append(
            {
                "projecao": name,
                "n": len(sample),
                "vizinhos_avaliacao": 10,
                "trustworthiness": float(trustworthiness(sample, embedding, n_neighbors=10)),
            }
        )
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(
        output / "tables/projection_diagnostics.csv", index=False, float_format="%.10g"
    )
    coordinates = pd.DataFrame({"source_row": train.iloc[positions].index, TARGET: labels})
    for name, embedding in embeddings.items():
        coordinates[f"{name}_1"], coordinates[f"{name}_2"] = embedding[:, 0], embedding[:, 1]
    coordinates.to_csv(output / "tables/projections.csv", index=False, float_format="%.10g")
    return pca, diagnostics, positions
