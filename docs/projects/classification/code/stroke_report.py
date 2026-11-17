"""Figuras e relatório derivados das tabelas, sem treinar ou selecionar modelos."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import read_json
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

COLORS = {"Dummy": "#8a94a6", "Logistic": "#167d9a", "RandomForest": "#b57617", "MLP": "#5b46ae"}


def table(frame, columns=None):
    if columns:
        frame = frame[columns]
    frame = frame.astype(object).where(frame.notna(), "ausente")
    return frame.to_markdown(index=False, floatfmt=".4f")


def finish(fig, output, filename):
    fig.tight_layout()
    fig.savefig(output / "figures" / filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(output, selected, evaluation):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    history = pd.read_csv(output / "tables/history_final.csv")
    primary = history[history.seed.eq(42)]
    best_epoch = evaluation["final_training"][0]["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, ylabel in zip(
        axes, ["loss", "ap"], ["BCE ponderada", "Average Precision"], strict=True
    ):
        ax.plot(primary.epoch, primary[f"train_{metric}"], label="Treino", color="#167d9a")
        ax.plot(primary.epoch, primary[f"val_{metric}"], label="Validação", color="#c06432")
        ax.axvline(best_epoch, ls="--", color="#454554", label=f"Checkpoint: época {best_epoch}")
        ax.set(xlabel="Época", ylabel=ylabel)
        ax.legend()
    fig.suptitle("Figura 1 — Aprendizado da MLP selecionada (seed 42; curvas em eval)")
    finish(fig, output, "01_learning.png")

    search = pd.read_csv(output / "tables/search.csv")
    ablation = search[search.id.isin(["c01", "c02", "c03", "c04"])].sort_values("id")
    fig, ax = plt.subplots(figsize=(8, 4.7))
    bars = ax.bar(
        ["Sem L2/dropout", "L2 = 0,001", "Dropout = 0,2", "L2 + dropout"],
        ablation.ap,
        color=["#8a94a6", "#167d9a", "#5b46ae", "#c06432"],
    )
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(
        xlabel="Regularização explícita (early stopping em todas)",
        ylabel="AP de validação",
        ylim=(0, ablation.ap.max() * 1.22),
        title="Figura 2 — Ablação: 24 → 32 → 1; lr=0,001; seed 42",
    )
    ax.legend([bars[0]], ["Mesmo split e inicialização das camadas"])
    finish(fig, output, "02_regularization.png")

    predictions = pd.read_csv(output / "tables/test_predictions.csv")
    test_metrics = pd.read_csv(output / "tables/test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for name in COLORS:
        data = predictions[predictions.model.eq(name) & predictions.seed.eq(42)]
        row = test_metrics[test_metrics.model.eq(name) & test_metrics.seed.eq(42)].iloc[0]
        precision, recall, _ = precision_recall_curve(data.y, data.score)
        axes[0].step(
            recall, precision, where="post", label=f"{name}: AP={row.ap:.3f}", color=COLORS[name]
        )
        fpr, tpr, _ = roc_curve(data.y, data.score)
        axes[1].plot(fpr, tpr, label=f"{name}: AUC={row.roc_auc:.3f}", color=COLORS[name])
    prevalence = evaluation["test_positive"] / evaluation["test_rows"]
    axes[0].axhline(prevalence, ls=":", color="#555555", label=f"Prevalência={prevalence:.3f}")
    axes[1].plot([0, 1], [0, 1], ls=":", color="#555555", label="Aleatório")
    axes[0].set(xlabel="Recall (classe 1)", ylabel="Precisão (classe 1)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(
        xlabel="Taxa de falsos positivos",
        ylabel="Taxa de verdadeiros positivos",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Figura 3 — Precision–recall e ROC no mesmo teste (seed 42)")
    finish(fig, output, "03_pr_roc.png")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, name in zip(axes, COLORS, strict=True):
        row = test_metrics[test_metrics.model.eq(name) & test_metrics.seed.eq(42)].iloc[0]
        ConfusionMatrixDisplay(
            np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int), display_labels=["0", "1"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} · t={row.threshold:.3f}")
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe observada")
    fig.suptitle("Figura 4 — Matrizes de confusão (limiares fixados na validação; contagem na cor)")
    finish(fig, output, "04_confusion.png")

    mlp = test_metrics[test_metrics.model.eq("MLP")]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.scatter(mlp.seed, mlp.ap, color=COLORS["MLP"], s=80, label="MLP por seed")
    ax.axhline(evaluation["mlp_ap_mean"], color=COLORS["MLP"], label="Média da MLP")
    ax.axhspan(
        evaluation["mlp_ap_mean"] - evaluation["mlp_ap_std"],
        evaluation["mlp_ap_mean"] + evaluation["mlp_ap_std"],
        color=COLORS["MLP"],
        alpha=0.12,
        label="Média ± desvio amostral",
    )
    ax.axhline(
        evaluation["baseline_ap"],
        color=COLORS[evaluation["baseline"]],
        ls="--",
        label="Baseline escolhido na validação",
    )
    ax.axhline(
        evaluation["baseline_ap"] * 1.1, color="#b57617", ls=":", label="Meta: baseline × 1,10"
    )
    ax.set(
        xticks=mlp.seed,
        xlabel="Seed de inicialização e batches (split fixo)",
        ylabel="AP no teste",
        title="Figura 5 — Variação entre três treinos independentes",
    )
    ax.legend(fontsize=8)
    finish(fig, output, "05_seeds.png")

    subgroups = pd.read_csv(output / "tables/subgroups.csv").sort_values("error_rate")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    labels = [f"{r.feature}={r.group} (n={r.n}; +={r.positive})" for r in subgroups.itertuples()]
    bars = ax.barh(labels, subgroups.error_rate * 100, color="#5b46ae")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    ax.set(
        xlabel="Erros / registros do subgrupo (%)",
        ylabel="Subgrupo (n e positivos observados)",
        xlim=(0, 90),
        title="Figura 6 — Erro por subgrupo no teste (MLP, seed 42)",
    )
    ax.legend([bars[0]], ["Taxa de classificação incorreta; grupos se sobrepõem"])
    finish(fig, output, "06_subgroups.png")

    importance = pd.read_csv(output / "tables/permutation.csv").sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        importance.feature,
        importance["mean"],
        xerr=importance["std"],
        color="#167d9a",
        capsize=3,
        label="Média ± desvio, 20 permutações",
    )
    ax.axvline(0, ls=":", color="#555555")
    ax.set(
        xlabel="Queda de AP na validação",
        ylabel="Variável original permutada",
        title="Figura 7 — Permutation importance (MLP, seed 42)",
    )
    ax.legend()
    finish(fig, output, "07_importance.png")


def build_report(output, protocol):
    selected = read_json(output / "tables/selection.json")
    result = read_json(output / "tables/evaluation.json")
    search = pd.read_csv(output / "tables/search.csv")
    baseline_val = pd.read_csv(output / "tables/baselines_validation.csv")
    tested = pd.read_csv(output / "tables/test_metrics.csv")
    by_class = pd.read_csv(output / "tables/test_per_class.csv")
    subgroups = pd.read_csv(output / "tables/subgroups.csv")
    importance = pd.read_csv(output / "tables/permutation.csv")
    worst = pd.read_csv(output / "tables/worst_errors.csv")
    history = pd.read_csv(output / "tables/history_final.csv").query("seed == 42")
    config = selected["config"]
    best = search.iloc[0]
    primary = tested.query("model == 'MLP' and seed == 42").iloc[0]
    final = result["final_training"][0]
    worst_group = subgroups.iloc[0]
    ablation = search[search.id.isin(["c01", "c02", "c03", "c04"])].sort_values("id")
    baseline_test = tested[tested.model.eq(selected["baseline"])].iloc[0]
    history_best = history.loc[history.epoch.eq(final["best_epoch"])].iloc[0]
    history_last = history.iloc[-1]
    summary = [
        [
            1,
            "Dataset, task, target and main metric",
            "Stroke Prediction v1; classificação binária; stroke; Average Precision (PR-AUC por degraus)",
        ],
        [
            2,
            "Train · validation · test (rows)",
            f"{selected['train_rows']} · {selected['validation_rows']} · {selected['test_rows']}",
        ],
        [
            3,
            "Best baseline and its test metric",
            f"{selected['baseline']} (escolhido na validação); AP teste={result['baseline_ap']:.6f}",
        ],
        [
            4,
            "Chosen network: layers × units, number of parameters, regularization",
            f"24 → {' → '.join(map(str, config['hidden']))} → 1; {final['parameters']} parâmetros; dropout={config['dropout']}; L2={config['weight_decay']}; early stopping",
        ],
        [
            5,
            "Optimizer, learning rate, batch and the epoch it stopped at",
            f"Adam; lr={config['lr']}; batch=128; parada={final['stopped_epoch']}, restaura época {final['best_epoch']} (seed 42)",
        ],
        [
            6,
            "The network's main metric on the test set (mean ± std, 3 seeds)",
            f"AP={result['mlp_ap_mean']:.6f} ± {result['mlp_ap_std']:.6f}; seeds 42,43,44; ddof=1",
        ],
        [
            7,
            "Success criterion met?",
            f"{'Sim' if result['success'] else 'Não'}; ganho relativo={result['relative_gain']:.2%}; meta=+10%; AP necessária={1.1 * result['baseline_ap']:.6f}",
        ],
        [
            8,
            "Subgroup with the largest error and its value",
            f"{worst_group.feature}={worst_group.group}; erro={worst_group.error_rate:.2%}; n={int(worst_group.n)}; seed 42",
        ],
    ]
    summary = pd.DataFrame(summary, columns=["#", "Summary", "Value"])
    summary.to_csv(output / "tables/results_summary.csv", index=False)
    build_figures(output, selected, result)
    architecture = " → ".join(map(str, [selected["n_features"], *config["hidden"], 1]))
    subgroup_comments = []
    hypotheses = {
        "age": "Idade domina o ranking: pode elevar alarmes nos mais velhos e ocultar AVC em jovens; associação observacional não é causalidade.",
        "hypertension": "A prevalência e a composição etária podem diferir; é preciso estudar interações e não atribuir o erro apenas à hipertensão.",
        "gender": "Diferenças de composição e poucos positivos podem explicar o contraste; estes números não demonstram equidade.",
        "bmi_missing": "A ausência pode atuar como marcador de coleta; associação com o alvo pode não se repetir em outra instituição.",
        "smoking_status": "Tabagismo, idade e preenchimento podem estar associados; a hipótese exige mais amostras e avaliação externa.",
    }
    for feature, group in subgroups.groupby("feature", sort=False):
        row = group.sort_values("error_rate", ascending=False).iloc[0]
        subgroup_comments.append(
            f"- **{feature}={row.group}:** {int(row.fp)} FP + {int(row.fn)} FN em {int(row.n)} registros = {row.error_rate:.2%} de erro; {int(row.positive)} positivos. Hipótese: {hypotheses[feature]}"
        )
    fn_examples, fp_examples = worst.query("y == 1"), worst.query("y == 0")
    examples_text = []
    for r in worst.itertuples():
        hypothesis = (
            "perfil jovem pode receber score baixo porque a idade domina o aprendizado"
            if r.y == 1
            else "idade elevada e/ou IMC ausente podem produzir um alarme mesmo sem AVC registrado"
        )
        examples_text.append(
            f"- Linha {r.source_row}: {'FN' if r.y == 1 else 'FP'}, idade {r.age:g}, score {r.score:.4f}, distância ao limiar {r.distance_to_threshold:.4f}. Hipótese: {hypothesis}."
        )
    comparison = "maior" if abs(result["gap"]) > result["mlp_ap_std"] else "menor"
    outcome = "atingiu" if result["success"] else "não atingiu"
    text = f"""---
project: classification
dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
team: ["João K Bresser"]
ai_use: "OpenAI Codex auxiliou implementação, execução dos experimentos, testes, figuras, análise e redação. Os números vêm das execuções e tabelas disponíveis."
---

# Projeto 2 — Classificação de registros de AVC com uma MLP

**Resultado:** a MLP obteve AP **{result["mlp_ap_mean"]:.4f} ± {result["mlp_ap_std"]:.4f}** no teste,
contra **{result["baseline_ap"]:.4f}** da regressão logística. A rede {outcome} o critério fixado
antes do treino. O ranking melhor na validação não se sustentou no teste.

A pergunta é se uma rede aprende relações que um modelo simples não captura. Mantemos
a tarefa da [EDA](../eda/index.md): identificar `stroke=1` no **Stroke Prediction v1**;
`0` significa ausência de AVC **registrado**. Sem datas, não demonstramos previsão de AVC futuro.

[Como reproduzir](reproduce.md) · [Guia para entender e explicar](study_guide.md) ·
[Protocolo definido antes do treinamento](protocol.json).

Referências do curso: [enunciado](https://insper.github.io/ann-dl/2026.2/projects/supervised/),
[conceitos](https://insper.github.io/ann-dl/2026.2/classes/concepts/),
[divisão dos dados](https://insper.github.io/ann-dl/2026.2/classes/data/splitting/),
[vazamento](https://insper.github.io/ann-dl/2026.2/classes/data/leakage/),
[desbalanceamento](https://insper.github.io/ann-dl/2026.2/classes/data/imbalance/),
[MLP](https://insper.github.io/ann-dl/2026.2/classes/mlp/),
[otimização](https://insper.github.io/ann-dl/2026.2/classes/optimization/),
[regularização](https://insper.github.io/ann-dl/2026.2/classes/regularization/),
[métricas](https://insper.github.io/ann-dl/2026.2/classes/metrics/classification/).

## 0. Starting point

### Continuidade da EDA e prevenção de vazamento

A base tem 5.110 registros e 10 atributos úteis. Importamos `make_preprocessor()` de
`docs/projects/eda/code/preprocessing.py`, em vez de copiar a implementação. O pipeline conserva:
mediana para numéricos ausentes, `log1p` para glicose/IMC, padronização das medidas,
indicador binário de IMC ausente e one-hot nas sete categorias. `Unknown` em tabagismo
permanece categoria. `id` e `stroke` não entram na matriz de atributos.

A divisão externa é a mesma da EDA: primeira dobra de
`StratifiedGroupKFold(5, shuffle=True, random_state=42)`, com grupos formados pelo hash
dos atributos. Conferimos todos os índices contra a [lista da EDA](../eda/tables/split.csv).
Os 4.088 registros de desenvolvimento são subdivididos pela mesma regra em
**{selected["train_rows"]} de treino e {selected["validation_rows"]} de validação**;
o **teste conserva {selected["test_rows"]} registros**. Nenhum perfil aparece em duas partições.
Há **{selected["train_positive"]} / {selected["validation_positive"]} / {result["test_positive"]} positivos**,
respectivamente. [Índices das três partições](tables/split.csv).

Imputador, scaler e vocabulário one-hot são **reajustados somente nas {selected["train_rows"]} linhas de treino**;
validação e teste recebem `transform`. A matriz tem **{selected["n_features"]} entradas**.
A mediana de IMC nesse treino é **{selected["train_bmi_median"]:.1f}**. Não usamos PCA/t-SNE/UMAP
como entradas: na EDA eram ferramentas exploratórias. Não recebemos uma devolutiva de correção
da EDA; não alegamos ter corrigido comentários inexistentes. A adaptação necessária é retirar
a validação antes de ajustar o pipeline.

### Métrica e meta estabelecidas antes do treino

A métrica principal é **Average Precision (AP)**, a integral por degraus da curva
precision–recall, calculada por `average_precision_score`; aqui esse é o significado de PR-AUC.
Com poucos positivos, acurácia alta pode ocorrer sem detectar um único AVC registrado.
AP avalia o ranking e não depende de um limiar. ROC-AUC e métricas de classe são complementares.

**Meta pré-fixada:** AP média da MLP nas três seeds no teste ≥ **1,10 × AP do baseline
escolhido pela maior AP de validação**. O ganho de 10% é um objetivo educacional de desempenho,
não um limiar de utilidade clínica. [Registro anterior ao treino](tables/protocol_before_training.json)
e [seleção congelada](tables/selection.json) registram o protocolo e hash do código.

O teste da EDA já havia passado por checagens de qualidade e transformação; não era desconhecido
em sentido absoluto. **Nenhuma métrica de classificador no teste foi usada nesta busca**.
A avaliação final ocorreu em lote depois de todos os treinos, limiares e configurações fixados.

## 1. Baselines

Os três modelos usam exatamente a mesma matriz do pipeline ajustado no treino.
`DummyClassifier(strategy="most_frequent")` sempre prevê a classe 0, com limiar 0,5.
A regressão logística usa `C=1`, `class_weight="balanced"` e até 3.000 iterações;
é a referência linear regularizada, sem busca de C. A floresta opcional tem 300 árvores,
`min_samples_leaf=5`, `balanced_subsample`, seed 42: captura não linearidades sem rede.
Seus parâmetros foram fixados antes da avaliação.

**Tabela 1 — Baselines na validação, ordenados por AP.**

{table(baseline_val, ["model", "ap", "roc_auc", "macro_f1", "precision_1", "recall_1", "f2_1", "threshold"])}

O baseline escolhido foi **{selected["baseline"]}**, AP **{baseline_val.iloc[0].ap:.4f}**.
Dummy tem acurácia **{baseline_val[baseline_val.model.eq("Dummy")].iloc[0].accuracy:.2%}**
e recall positivo zero. Portanto, acurácia não guia a escolha.

Os limiares dos modelos não triviais maximizam **F2 apenas na validação**. Como
F2 = 5TP/(5TP + 4FN + FP), a regra valoriza reduzir FN, aceitando mais FP;
isso explicita uma preferência educacional por recall. Não estimamos custos clínicos reais.
Empates usam o maior limiar. Scores com ponderação de classe **não são probabilidades calibradas**.

## 2. The network

A busca comparou camadas ocultas de 16, 32, 32–16 e 64–32 unidades. A escolhida foi
**{config["id"]}: {architecture}**, com ReLU nas ocultas e **uma saída linear (logit)**.
São **{final["parameters"]} parâmetros**, contra **{selected["train_rows"]} linhas de treino**,
razão **{final["parameters"] / selected["train_rows"]:.3f} parâmetro/linha**, mas só
{selected["train_positive"]} positivos. Essa razão não garante generalização.
Uma MLP pequena é proporcional ao porte e permite interações sem supor que tabelas
exijam redes profundas. A seleção favoreceu a menor largura testada, pela AP de validação.

As camadas ocultas usam inicialização **Kaiming normal**, apropriada à ReLU, e biases zero;
a saída usa **Xavier normal**. A loss é `BCEWithLogitsLoss`, que combina sigmoid e
entropia cruzada binária de modo numericamente estável. O `forward` **não** aplica sigmoid;
ela é aplicada uma única vez para produzir scores na inferência.

O desbalanceamento é tratado por **pos_weight = negativos/positivos = {final["pos_weight"]:.4f}**,
calculado somente no treino e mantido na validação. Não há reamostragem de validação/teste.
Esse peso aumenta a contribuição dos raros positivos no gradiente; não cria informação nova.

**Arquitetura executável:**

```python
--8<-- "docs/projects/classification/code/network.py"
```

## 3. Training

Usamos **Adam**, `lr={config["lr"]}`, betas padrão (0,9; 0,999), batches de **128**,
até **200 épocas**. Adam adapta a escala das atualizações; 0,001 é o ponto de partida
para MLP tabular apresentado na aula. A busca testa também 0,0003 e 0,003.
O batch equilibra atualizações frequentes e custo nesta base pequena; não foi ajustado pelo teste.
O treinamento roda em CPU, uma thread, com seeds de Python, NumPy e PyTorch fixadas.

Em cada batch: zerar gradientes, forward, calcular BCE, `backward()` e `optimizer.step()`.
Depois de cada época, `eval()` desativa dropout; treino e validação são medidos com a mesma BCE
ponderada, sem incluir a penalidade L2 na curva. `no_grad()` evita construir o grafo na avaliação.

O **early stopping** monitora BCE de validação, paciência **25 épocas** e melhora mínima
**0,0001**. O checkpoint de menor BCE aceito por esse critério é restaurado antes de calcular
AP e escolher o limiar. A AP seleciona a configuração; a BCE seleciona a época.
Esses objetivos não são iguais, e por isso as duas curvas são mostradas.

![Figura 1 — Curvas de loss e AP](figures/01_learning.png)

Na seed 42, o melhor checkpoint é a **época {final["best_epoch"]}**:
BCE treino **{history_best.train_loss:.4f}**, validação **{history_best.val_loss:.4f}**,
AP treino **{history_best.train_ap:.4f}**, AP validação **{history_best.val_ap:.4f}**.
A execução para na época **{final["stopped_epoch"]}**; na última época a BCE de treino é
**{history_last.train_loss:.4f}** e a de validação **{history_last.val_loss:.4f}**.
O período após a melhor época indica que continuar reduzindo o erro de treino não compra melhora
de validação sustentada; é o sinal operacional de sobreajuste usado para parar.
A convergência é operacional, não prova de ótimo global.

**Loop executado:**

```python
--8<-- "docs/projects/classification/code/training.py"
```

## 4. Regularization and hyperparameters

Testamos **12 configurações** previamente listadas em `protocol.json`, todas com seed 42.
As quatro primeiras isolam duas técnicas na mesma arquitetura 24–32–1 e lr 0,001:
**L2 via `Adam(weight_decay=0.001)`**, **dropout 0,2**, ambos e nenhuma regularização explícita.
L2 penaliza pesos grandes; dropout remove ativações aleatoriamente apenas no treino.
Aqui é Adam com L2 acoplado, não AdamW. **Early stopping é comum a todas**;
“sem regularização” nesta ablação significa sem L2 e sem dropout, não sem parada antecipada.

**Tabela 2 — Ablação controlada na validação.**

{table(ablation, ["id", "dropout", "weight_decay", "ap", "best_epoch", "stopped_epoch"])}

![Figura 2 — Comparação de regularização](figures/02_regularization.png)

A AP sem L2/dropout foi **{ablation.iloc[0].ap:.4f}**; com L2, **{ablation.iloc[1].ap:.4f}**;
com dropout, **{ablation.iloc[2].ap:.4f}**; com ambos, **{ablation.iloc[3].ap:.4f}**.
Regularização não oferece melhora automática: muda o equilíbrio entre ajuste e generalização.
Essa ablação usa uma seed, portanto não isola a variância entre inicializações.

**Tabela 3 — Busca completa ordenada por AP de validação (maior é melhor).**

{table(search, ["rank", "id", "hidden", "lr", "dropout", "weight_decay", "parameters", "ap", "best_epoch", "stopped_epoch"])}

Escolhemos **{config["id"]}**, AP **{best.ap:.4f}**, antes de abrir o teste.
Não adicionamos uma configuração depois de observar o resultado final.
[Busca completa](tables/search.csv) · [Histórico por época](tables/history_search.csv).

## 5. Test evaluation

A arquitetura escolhida foi **retreinada do zero nas seeds 42, 43 e 44**.
As três usam o mesmo treino/validação; cada uma restaura seu checkpoint e escolhe seu limiar
na validação. Mantemos a validação separada para early stopping e limiar, portanto
**não** juntamos as 4.088 linhas para o ajuste final. Os baselines também são reajustados
no mesmo treino. Nenhuma seed foi escolhida pelo teste e não fazemos ensemble.
A seed 42 é a referência das figuras e análise de erros, fixada no protocolo.

[Registro dos treinos e limiares antes do teste](tables/final_before_test.json) ·
[Avaliação e versões](tables/evaluation.json) · [Predições auditáveis](tables/test_predictions.csv).

**Tabela 4 — Todos os modelos no mesmo teste. AP/ROC não dependem do limiar.**

{table(tested, ["model", "seed", "ap", "roc_auc", "macro_f1", "precision_1", "recall_1", "f1_1", "threshold"])}

**Tabela 5 — Precisão, recall e F1 por classe (seed 42).**

{table(by_class[by_class.seed.eq(42)], ["model", "class", "precision", "recall", "f1-score", "support"])}

As demais seeds também têm métricas por classe no [CSV completo](tables/test_per_class.csv).
O teste tem **{result["test_positive"]} positivos/{result["test_rows"]} registros**, prevalência
**{result["test_positive"] / result["test_rows"]:.2%}**. A AP de Dummy é essa prevalência, enquanto
ROC-AUC de Dummy é 0,5. A rede aprende sinal acima da referência trivial, mas isso não
demonstra vantagem sobre o modelo linear.

![Figura 3 — Curvas precision–recall e ROC](figures/03_pr_roc.png)

Na seed 42, AP **{primary.ap:.4f}** e ROC-AUC **{primary.roc_auc:.4f}** resumem rankings diferentes;
a PR expõe melhor a baixa precisão entre os alarmes nesta base rara.

![Figura 4 — Matrizes de confusão](figures/04_confusion.png)

No limiar **{primary.threshold:.6f}**, a rede de referência tem **{int(primary.tp)} TP,
{int(primary.fp)} FP, {int(primary.fn)} FN e {int(primary.tn)} TN**.
Precisão positiva **{primary.precision_1:.2%}**, recall **{primary.recall_1:.2%}**,
F1 positivo **{primary.f1_1:.4f}** e macro F1 **{primary.macro_f1:.4f}**.
A logística tem **{int(baseline_test.tp)} TP e {int(baseline_test.fp)} FP** com seu próprio
limiar de validação; não comparamos usando limiar ajustado no teste.

![Figura 5 — Resultado nas três seeds](figures/05_seeds.png)

A AP da MLP é **{result["mlp_ap_mean"]:.6f} ± {result["mlp_ap_std"]:.6f}**
(média ± desvio **amostral**, `ddof=1`, três seeds). A diferença média para o baseline é
**{result["gap"]:+.6f}**, magnitude **{comparison}** que o desvio entre seeds.
Isso não é um teste de significância: mede só variação de inicialização e batches,
não incerteza amostral, deslocamento de domínio ou variância da divisão dos dados.
Com apenas 50 positivos no teste, um único FN altera o recall em 2 pontos percentuais.

O ganho relativo é **{result["relative_gain"]:.2%}**. A meta exigiria AP ≥
**{1.1 * result["baseline_ap"]:.6f}**; **{"foi atingida" if result["success"] else "não foi atingida"}**.
Comparar 12 candidatas em uma validação com somente 40 positivos também pode selecionar
uma configuração favorecida pelo ruído. A queda no teste é compatível com esse problema;
não prova, por si só, uma causa única.

## 6. Error analysis

### Subgrupos definidos antes do teste

A taxa de erro é **(FP+FN)/n do subgrupo** na MLP seed 42. Ela é diferente de
taxa de falsos negativos: um grupo quase todo negativo pode ter erro baixo e recall zero.
Os grupos se sobrepõem; não se somam. Mostramos n e número de positivos para não ocultar suporte.

**Tabela 6 — Erros e recall por subgrupo.**

{table(subgroups, ["feature", "group", "n", "positive", "error_rate", "fp", "fn", "recall_1"])}

![Figura 6 — Erro por subgrupo](figures/06_subgroups.png)

{chr(10).join(subgroup_comments)}

Entre menores de 45 anos, o recall é **0/5 = 0%**, apesar de erro global baixo nesse grupo.
Entre 45–64 anos, o recall é **4/16 = 25%**; entre ≥65, é **28/29 = 96,55%**,
com **152 falsos positivos**. A forte preferência por idade ajuda a explicar os dois extremos.
Não há `gender=Other` no teste; não estimamos seu desempenho. Nenhuma conclusão robusta
sobre grupos raros ou equidade decorre deste split.

### Piores exemplos

Selecionamos cinco FN e cinco FP pela maior distância ao limiar no lado incorreto.
Isso mede o afastamento da decisão, **não confiança probabilística calibrada**.
`source_row` é o índice original sem cabeçalho; o identificador `id` continua excluído.

**Tabela 7 — Piores erros de cada tipo (MLP seed 42).**

{table(worst, ["source_row", "y", "score", "age", "avg_glucose_level", "bmi", "hypertension"])}

Os cinco FN têm idades de **{fn_examples.age.min():g}–{fn_examples.age.max():g}** anos;
os cinco FP, **{fp_examples.age.min():g}–{fp_examples.age.max():g}** anos, e
**{int(fp_examples.bmi.isna().sum())}/5** deles têm IMC ausente. Hipóteses para cada exemplo:

{chr(10).join(examples_text)}

Essas hipóteses descrevem falhas do modelo; não explicam a ocorrência individual de AVC.
Podem existir atributos ausentes ou problemas de cronologia que esta base não permite investigar.

### Permutation importance e diálogo com a EDA

Na **validação**, permutamos cada variável original **20 vezes**, recalculando a AP
sem reajustar o modelo nem o pipeline. Para `bmi`, valor e ausência são permutados juntos
antes da transformação; para categorias, todas as colunas one-hot mudam de forma coerente.
Usar validação aqui preserva o teste para avaliação, sem novas decisões a partir de suas métricas.

**Tabela 8 — Importância por queda de AP na validação.**

{table(importance)}

![Figura 7 — Permutation importance](figures/07_importance.png)

A maior queda vem de **{importance.iloc[0].feature}: {importance.iloc[0]["mean"]:.4f} ± {importance.iloc[0]["std"]:.4f}**.
Idade dominante e a contribuição de glicose/IMC são compatíveis com a EDA, que mostrou
associações e sobreposição de classes. O efeito quase nulo de hipertensão, apesar da
associação bruta, lembra que atributos correlacionados podem carregar informação redundante.
Importância negativa ou próxima de zero pode ser ruído ou substituição por outro atributo;
não equivale a ausência de associação, efeito causal ou irrelevância clínica.
Os desvios são de permutações no mesmo conjunto, não intervalos de confiança populacional.

## 7. Conclusion

**A rede não compensou como melhoria de AP sobre a regressão logística neste experimento.**
Ela superou Dummy e ganhou na validação, mas sua média no teste ficou abaixo do baseline linear,
sem atingir a meta de 10%. O resultado sustenta manter a logística como referência mais simples
para novos estudos, sem declarar que redes nunca funcionem nesta base.

Limitações: poucos positivos; uma única divisão de validação; busca com 12 candidatas;
origem confidencial; ausência de datas e de população amostral definida; pesos de classe sem
calibração; dependência do padrão de ausência de IMC e da idade; sem validação externa.
A EDA já havia observado algumas propriedades do teste, embora não métricas preditivas.

Um falso negativo deixaria um registro positivo sem identificação; um falso positivo poderia
gerar alarme e trabalho adicional se alguém tentasse usar a saída para triagem. Os erros
não têm impacto real neste estudo educacional. Não há validação que sustente aplicação clínica.

Com mais tempo: validação cruzada estratificada por grupos **dentro do desenvolvimento**, pipeline
ajustado em cada fold, comparação de rankings por folds, avaliação/calibração em dados próprios
e obtenção de uma nova base com cronologia e origem verificáveis. O teste atual não serviria
para escolher essas mudanças; uma avaliação futura exigiria um novo conjunto externo.

### Rastreabilidade e conteúdos das aulas

O estudo usa aprendizado supervisionado (aula 1), tratamento dos dados e divisão (aula 2),
neurônios e composição de camadas (aulas 3–5), gradientes e Adam (aula 6), L2/dropout/early stopping
(aula 7), AP/ROC/F2 e confusão (aula 8). O [guia de estudo](study_guide.md) conecta cada conceito
ao código e às decisões. Não precisamos de CNN, atenção ou modelos generativos para estes dados tabulares.

Os resultados são reproduzíveis pelos arquivos em `code/`; as tabelas e figuras foram geradas
por execução, não preenchidas com números hipotéticos. O uso de IA está declarado no cabeçalho.

## Results summary

{table(summary)}
"""
    (output / "index.md").write_text(text)
    print(f"Relatório e sete figuras gravados em {output}", flush=True)
