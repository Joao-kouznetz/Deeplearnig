"""Estatísticas e figuras da EDA; somente os dados de treino entram nas etapas 2–4."""

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
from dataset import (
    ALL_NUMERIC,
    CATEGORICAL,
    ENGAGEMENT,
    MONTHS,
    PROJECT,
    SEED,
    TARGET,
    profile_groups,
)
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from umap import UMAP

COLORS = ["#416788", "#C65D21"]
CLASS_NAMES = ["Sem compra", "Com compra"]
LABELS = [
    Line2D([0], [0], marker="o", linestyle="", color=c, label=n)
    for c, n in zip(COLORS, CLASS_NAMES, strict=True)
]


def save_figure(fig, name, output=PROJECT):
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(
        output / "figures" / f"{name}.png",
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "Deeplearnig EDA"},
    )
    plt.close(fig)


def ordered_values(frame, column):
    if column == "Month":
        return [m for m in MONTHS if m in set(frame[column])]
    return sorted(frame[column].unique(), key=lambda x: (len(x), x))


def wilson(success, total):
    z = 1.96
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return center - half, center + half


def descriptive_tables(raw, clean, train, test, output=PROJECT):
    tables = {}
    quality = pd.DataFrame(
        {
            "ausentes_n": raw.isna().sum(),
            "ausentes_pct": raw.isna().mean() * 100,
            "cardinalidade": raw.nunique(),
        }
    )
    quality.index.name = "variavel"
    tables["quality"] = quality
    numeric = train[ALL_NUMERIC].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
    numeric["skewness"] = train[ALL_NUMERIC].skew()
    numeric["zero_pct"] = train[ALL_NUMERIC].eq(0).mean() * 100
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
    corr = train[ALL_NUMERIC].corr(method="spearman")
    tables["correlation"] = corr
    pairs = [
        {"variavel_1": a, "variavel_2": b, "rho": float(corr.loc[a, b])}
        for i, a in enumerate(ALL_NUMERIC)
        for b in ALL_NUMERIC[i + 1 :]
    ]
    tables["correlation_pairs"] = pd.DataFrame(pairs).sort_values(
        "rho", key=lambda s: s.abs(), ascending=False
    )
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
                    "compras": int(values.sum()),
                    "taxa_pct": values.mean() * 100,
                    "wilson_low_pct": low * 100,
                    "wilson_high_pct": high * 100,
                }
            )
    tables["category_target"] = pd.DataFrame(conditional)
    box_rows = []
    for visitor, group in train.groupby("VisitorType"):
        for col in ["ProductRelated", "ExitRates"]:
            box_rows.append(
                {
                    "VisitorType": visitor,
                    "variavel": col,
                    "n": len(group),
                    "q1": group[col].quantile(0.25),
                    "mediana": group[col].median(),
                    "q3": group[col].quantile(0.75),
                    "IQR": group[col].quantile(0.75) - group[col].quantile(0.25),
                }
            )
    tables["grouped_numeric"] = pd.DataFrame(box_rows)
    q1, q3 = train[ENGAGEMENT].quantile(0.25), train[ENGAGEMENT].quantile(0.75)
    thresholds = q3 + 1.5 * (q3 - q1)
    flags = train[ENGAGEMENT].gt(thresholds)
    tables["outliers"] = pd.DataFrame(
        {"q1": q1, "q3": q3, "limiar_superior": thresholds, "acima_limiar_n": flags.sum()}
    )
    # Regras de domínio definidas antes de observar os dados; não excluir caudas válidas.
    invalid = []
    for col in ALL_NUMERIC:
        invalid.append({"regra": f"{col} < 0", "n": int(raw[col].lt(0).sum())})
    for col in ["BounceRates", "ExitRates", "SpecialDay"]:
        invalid.append({"regra": f"{col} > 1", "n": int(raw[col].gt(1).sum())})
    for col in ["Administrative", "Informational", "ProductRelated"]:
        invalid.append({"regra": f"{col} não inteiro", "n": int(raw[col].mod(1).ne(0).sum())})
    for pages, duration in zip(ENGAGEMENT[::2], ENGAGEMENT[1::2], strict=True):
        invalid.append(
            {
                "regra": f"{pages}=0 e {duration}>0",
                "n": int((raw[pages].eq(0) & raw[duration].gt(0)).sum()),
            }
        )
    tables["domain_checks"] = pd.DataFrame(invalid)
    classes = pd.DataFrame(
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
    tables["classes"] = classes
    membership = pd.concat(
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
    tables["split"] = membership
    tables["summary_counts"] = {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "duplicates_removed": len(raw) - len(clean),
        "train_rows": len(train),
        "test_rows": len(test),
        "profile_groups": int(len(np.unique(profile_groups(clean)))),
        "repeated_profile_rows": int(len(clean) - len(np.unique(profile_groups(clean)))),
        "outlier_train_rows": int(flags.any(axis=1).sum()),
        "log_changed_train_rows": int(train[ENGAGEMENT].gt(0).any(axis=1).sum()),
        "missing_total": int(raw.isna().sum().sum()),
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
    ax.set(xlabel="Desfecho da sessão", ylabel="Sessões", ylim=(0, counts.max() * 1.2))
    ax.set_title("Figura 1 — Distribuição de Revenue no arquivo original")
    ax.legend()
    save_figure(fig, "01_target", output)

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    for ax, col in zip(axes.flat, ALL_NUMERIC, strict=True):
        # Bins em log1p apenas para visualização das caudas, sem alterar o treino.
        if col in ENGAGEMENT + ["PageValues"]:
            bins = np.expm1(np.linspace(0, np.log1p(train[col].max()), 32))
            ax.set_xscale("symlog", linthresh=1)
        else:
            bins = 30
        ax.hist(train[col], bins=bins, color=COLORS[0], label="Treino")
        ax.set(
            title=f"{col} | assimetria={train[col].skew():.2f}",
            xlabel=col + (" (symlog)" if col in ENGAGEMENT + ["PageValues"] else " (fração)"),
            ylabel="Sessões",
        )
        ax.legend()
    fig.suptitle("Figura 2 — Distribuições numéricas (treino; PageValues somente para auditoria)")
    save_figure(fig, "02_numeric", output)

    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    for ax, col in zip(axes.flat, CATEGORICAL, strict=False):
        freq = train[col].value_counts(normalize=True).reindex(ordered_values(train, col)) * 100
        ax.bar(freq.index, freq, color=COLORS[0], label="Treino")
        ax.set(title=col, xlabel="Categoria", ylabel="Sessões (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    axes.flat[-1].set_visible(False)
    fig.suptitle("Figura 3 — Frequências das sete variáveis categóricas (treino)")
    save_figure(fig, "03_categorical", output)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        tables["correlation"],
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Correlação de Spearman (ρ)"},
    )
    ax.set(
        xlabel="Variável numérica",
        ylabel="Variável numérica",
        title="Figura 4 — Correlações monotônicas (treino; PageValues em auditoria)",
    )
    save_figure(fig, "04_correlation", output)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (a, b) in zip(
        axes,
        [("ProductRelated", "ProductRelated_Duration"), ("BounceRates", "ExitRates")],
        strict=True,
    ):
        for cls, color, label in zip([0, 1], COLORS, CLASS_NAMES, strict=True):
            group = train[train[TARGET].eq(cls)]
            ax.scatter(group[a], group[b], s=8, alpha=0.3, c=color, label=label, rasterized=True)
        ax.set(xlabel=a, ylabel=b, title=f"ρ = {tables['correlation'].loc[a, b]:.3f}")
        if a == "ProductRelated":
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
        ax.legend()
    fig.suptitle("Figura 5 — Dois pares associados, coloridos pelo alvo (treino)")
    save_figure(fig, "05_scatter", output)

    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    for ax, col in zip(axes.flat, CATEGORICAL, strict=False):
        tab = tables["category_target"].query("variavel == @col")
        x = np.arange(len(tab))
        ax.vlines(
            x, tab.wilson_low_pct, tab.wilson_high_pct, color=COLORS[0], label="IC Wilson 95%"
        )
        ax.scatter(x, tab.taxa_pct, color=COLORS[1], label="Compra observada", zorder=3)
        ax.axhline(
            train[TARGET].mean() * 100, color="gray", linestyle="--", label="Média do treino"
        )
        ax.set(
            xticks=x,
            xticklabels=tab.categoria,
            xlabel="Categoria",
            ylabel="Sessões com compra (%)",
            title=col,
            ylim=(0, max(40, tab.wilson_high_pct.max() + 3)),
        )
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Figura 6 — Compra por categoria, com incerteza amostral (treino)")
    save_figure(fig, "06_category_target", output)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ["ProductRelated", "ExitRates"], strict=True):
        sns.boxplot(
            data=train,
            x="VisitorType",
            y=col,
            hue=TARGET,
            palette=COLORS,
            order=ordered_values(train, "VisitorType"),
            showfliers=False,
            ax=ax,
        )
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, CLASS_NAMES, title="Revenue")
        ax.set(xlabel="Tipo de visitante", ylabel=col)
        if col == "ProductRelated":
            ax.set_yscale("symlog", linthresh=1)
    fig.suptitle("Figura 7 — Localização e dispersão por visitante e alvo (treino; caudas ocultas)")
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
        f"Figura 9 — Mesmas {len(sample):,} sessões de treino em cinco projeções 2D (seed=42)"
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
    coordinates = pd.DataFrame({"source_row": train.iloc[positions].index, "Revenue": labels})
    for name, embedding in embeddings.items():
        coordinates[f"{name}_1"], coordinates[f"{name}_2"] = embedding[:, 0], embedding[:, 1]
    coordinates.to_csv(output / "tables/projections.csv", index=False, float_format="%.10g")
    return pca, diagnostics, positions
