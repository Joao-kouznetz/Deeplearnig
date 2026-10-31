"""Relatório da base indicada no enunciado, com valores calculados pelo código."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataset import CATEGORICAL, FEATURES, NUMERIC, PROJECT, SHA256, SOURCE, TARGET

DICTIONARY = [
    ("id", "Identificador do registro; descartado", "Identificador", "código"),
    ("gender", "Categoria de gênero registrada", "Categórica nominal", "Male / Female / Other"),
    ("age", "Idade; valores fracionários também ocorrem", "Numérica contínua", "anos"),
    ("hypertension", "Hipertensão registrada", "Categórica binária", "0 = não; 1 = sim"),
    ("heart_disease", "Doença cardíaca registrada", "Categórica binária", "0 = não; 1 = sim"),
    ("ever_married", "Se a pessoa já foi casada", "Categórica binária", "No / Yes"),
    (
        "work_type",
        "Tipo de trabalho; children e Never_worked são categorias",
        "Categórica nominal",
        "categoria",
    ),
    ("Residence_type", "Residência rural ou urbana", "Categórica binária", "Rural / Urban"),
    (
        "avg_glucose_level",
        "Glicose média registrada",
        "Numérica contínua",
        "unidade não documentada",
    ),
    ("bmi", "Índice de massa corporal (IMC)", "Numérica contínua", "kg/m² (definição de IMC)"),
    (
        "smoking_status",
        "Tabagismo; Unknown significa informação indisponível",
        "Categórica nominal",
        "categoria",
    ),
    ("stroke", "Registro de AVC; alvo da classificação", "Categórica binária", "0 = não; 1 = sim"),
]


def num(value):
    return f"{int(value):,}".replace(",", ".")


def pct(value):
    return f"{value:.2f}%".replace(".", ",")


def md(table, index=False, precision=3):
    return table.to_markdown(index=index, floatfmt=f".{precision}f")


def report(
    raw,
    clean,
    train,
    test,
    tables,
    pca,
    diagnostics,
    feature_names,
    train_matrix,
    test_matrix,
    checks,
    output=PROJECT,
):
    counts = tables["summary_counts"]
    stats = tables["numeric"]
    cats = tables["categorical"]
    conditional = tables["category_target"]
    pair = tables["correlation_pairs"].iloc[0]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pc90 = int(np.searchsorted(cumulative, 0.9) + 1)
    dictionary = pd.DataFrame(
        DICTIONARY, columns=["Variável", "Significado", "Tipo semântico", "Unidade"]
    )
    description = stats[
        ["count", "missing_n", "mean", "50%", "std", "min", "25%", "75%", "99%", "max", "skewness"]
    ].copy()
    description.columns = [
        "n observado",
        "ausentes",
        "média",
        "mediana",
        "desvio",
        "mín",
        "Q1",
        "Q3",
        "P99",
        "máx",
        "assimetria",
    ]
    numerical_notes = []
    for col in NUMERIC:
        row = stats.loc[col]
        frequencies, edges = np.histogram(train[col].dropna(), bins=30)
        peak = int(frequencies.argmax())
        interpretation = (
            "Assimetria pequena; idade não é tomada como distribuição normal."
            if col == "age"
            else "A média excede a mediana e há cauda à direita; log1p comprime essa cauda."
        )
        numerical_notes.append(
            f"- **{col}:** média {row['mean']:.3f}, mediana {row['50%']:.3f}, "
            f"assimetria {row.skewness:.3f}, máximo {row['max']:.3f}. {interpretation} "
            f"O bin mais frequente é [{edges[peak]:.3f}, {edges[peak + 1]:.3f}], "
            f"com {int(frequencies[peak])} observações; isso é uma moda por intervalo, não uma moda exata."
        )
    categorical_notes = []
    for col in CATEGORICAL:
        tab = cats[cats.variavel.eq(col)]
        majority = tab.iloc[0]
        rare = tab.loc[tab.rara_menos_1pct, "categoria"].astype(str).tolist()
        categorical_notes.append(
            f"- **{col}:** {len(tab)} categorias; mais frequente `{majority.categoria}` "
            f"(n={num(majority.n)}; {pct(majority.pct)}). Categorias com frequência <1%: "
            f"{', '.join(f'`{c}`' for c in rare) or 'nenhuma'}."
        )
    relation_notes = []
    for col in CATEGORICAL:
        tab = conditional[conditional.variavel.eq(col) & conditional.n.ge(100)]
        low, high = tab.loc[tab.taxa_pct.idxmin()], tab.loc[tab.taxa_pct.idxmax()]
        relation_notes.append(
            f"- **{col}:** entre níveis com n≥100, {pct(low.taxa_pct)} em `{low.categoria}` "
            f"(n={num(low.n)}) a {pct(high.taxa_pct)} em `{high.categoria}` (n={num(high.n)})."
        )
    weights = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=["PC1", "PC2"])
    weights.index.name = "feature"
    weights.to_csv(output / "tables/pca_weights.csv", float_format="%.10g")
    top_weights = pd.DataFrame(
        [
            {"componente": component, "feature": name, "peso": weights.loc[name, component]}
            for component in ["PC1", "PC2"]
            for name in weights[component].abs().nlargest(5).index
        ]
    )
    top_weights.to_csv(output / "tables/pca_top_weights.csv", index=False, float_format="%.10g")
    component_notes = []
    for component in ["PC1", "PC2"]:
        prominent = weights[component].abs().nlargest(3).index
        same = [name for name in prominent if weights.loc[name, component] >= 0]
        opposite = [name for name in prominent if weights.loc[name, component] < 0]
        component_notes.append(
            f"- **{component}:** maiores pesos positivos entre os três principais: "
            f"{', '.join(f'`{s}`' for s in same) or 'nenhum'}; negativos: "
            f"{', '.join(f'`{s}`' for s in opposite) or 'nenhum'}. "
            "Pesos do mesmo sinal contribuem no mesmo sentido para a coordenada; sinais opostos se contrapõem. Isso não prova correlação entre as variáveis originais."
        )
    grouped = tables["grouped_numeric"]
    box_notes = []
    for col in ["age", "avg_glucose_level"]:
        rows = grouped[grouped.variavel.eq(col)]
        box_notes.append(
            f"- **{col}:** "
            + "; ".join(
                f"hipertensão={r.hypertension}: mediana {r.mediana:.3f}, IQR {r.IQR:.3f}"
                for r in rows.itertuples()
            )
            + "."
        )
    summary = pd.DataFrame(
        [
            [
                1,
                "Dataset, tarefa e alvo",
                "Stroke Prediction (fedesoriano, Kaggle, v1); classificação; stroke",
            ],
            [
                2,
                "Instâncias × atributos (numéricos / categóricos)",
                f"{num(len(raw))} × 12 colunas no CSV; 10 atributos (3 / 7), mais id e alvo; id descartado",
            ],
            [
                3,
                "Coluna com mais ausentes e percentual",
                f"Ausência declarada: smoking_status, {num(counts['unknown_smoking_rows'])} Unknown ({pct(raw.smoking_status.eq('Unknown').mean() * 100)}); NaN: bmi, 201 ({pct(raw.bmi.isna().mean() * 100)})",
            ],
            [4, "Colunas descartadas e motivo", "id: identificador; stroke separado como alvo"],
            [
                5,
                "Classe minoritária (%)",
                f"stroke=1: {num(clean[TARGET].sum())} ({pct(clean[TARGET].mean() * 100)})",
            ],
            [
                6,
                "Tamanhos de treino e teste",
                f"{num(len(train))} / {num(len(test))}; seed=42; estratificação por perfis",
            ],
            [
                7,
                "Par numérico mais correlacionado e valor",
                f"{pair.variavel_1} × {pair.variavel_2}; Spearman ρ={pair.rho:.6f}; n={int(pair.n_pares_completos)}",
            ],
            [
                8,
                "Linhas afetadas pela estratégia de outliers",
                f"Treino: {num(counts['outlier_train_rows'])} sinalizadas por IQR; 0 removidas/limitadas; {num(counts['log_changed_train_rows'])} alteradas por log1p",
            ],
            [9, "Variância explicada por PC1 + PC2", pct(cumulative[1] * 100)],
            [
                10,
                "Shape de treino e teste após pipeline",
                f"{train_matrix.shape} / {test_matrix.shape}; 0 NaN e 0 infinitos",
            ],
        ],
        columns=["#", "Resumo dos resultados", "Valor"],
    )
    summary.to_csv(output / "tables/results_summary.csv", index=False)
    names_table = pd.DataFrame(
        {"posição (0-based)": range(len(feature_names)), "feature": feature_names}
    )
    text = f"""---
project: eda
task: classification
dataset: {SOURCE}
team:
  - João K Bresser
ai_use: "Projeto feito usando inteligência artificial (OpenAI Codex): código, análise, gráficos, testes e redação. Base selecionada pelo assistente entre os exemplos do enunciado; escolha específica ainda sujeita à confirmação do aluno. Resultados calculados pela execução do código."
---

# Projeto 1 — Análise exploratória de registros de AVC

**Pergunta:** quais características registradas se associam ao indicador `stroke`?
A tarefa é classificação binária: 1 indica AVC registrado e 0 indica ausência desse registro.
O objetivo desta entrega é entender os dados e preparar um pipeline para uma futura rede neural;
nenhum classificador é treinado e nenhuma métrica de desempenho preditivo é alegada.

A base **Stroke Prediction** é um dos exemplos de classificação explicitamente linkados no
[enunciado](https://insper.github.io/ann-dl/2026.2/projects/eda/#0-proposal). Ela foi selecionada pelo assistente
entre os exemplos; a escolha específica ainda pode ser confirmada ou alterada pelo aluno.
[Conferência com as aulas e exercícios](course_alignment.md).
**Uso de IA:** implementação, redação e interpretações foram produzidas com OpenAI Codex;
os números são resultados de execução, e a revisão e compreensão pelo aluno continuam necessárias.

**Escopo:** estudo educacional e observacional. Não há data de coleta, instante do AVC ou horizonte
preditivo: não interpretamos associações como risco futuro, causalidade ou recomendação clínica.

Referências: [enunciado e rubrica](https://insper.github.io/ann-dl/2026.2/projects/eda/),
[formato de entrega](https://insper.github.io/ann-dl/2026.2/exercises/submission/),
[fonte dos dados]({SOURCE}),
[metadados oficiais do Kaggle](https://www.kaggle.com/api/v1/datasets/view/fedesoriano/stroke-prediction-dataset),
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
e [parâmetros do UMAP](https://umap-learn.readthedocs.io/en/latest/parameters.html).

## 0. Proposal

**Base escolhida: Stroke Prediction Dataset, de fedesoriano, Kaggle, versão 1.**
O arquivo tem **{num(len(raw))} linhas × 12 colunas**, correspondendo a **10 atributos úteis
(3 numéricos e 7 categóricos)**, um identificador e o alvo. Atende aos requisitos de porte, tipos
mistos, disponibilidade pública e alvo presente no arquivo.

A base permite estudar desbalanceamento forte, dados ausentes e associações entre idade,
características registradas e o indicador de AVC. A combinação de variáveis numéricas e categóricas
é adequada para investigar preparação de entradas para uma rede neural. O primeiro risco é a
origem confidencial e a ausência de datas, que impedem verificar disponibilidade dos atributos
antes do desfecho ou representatividade da população.

[Proposta completa e vínculo com o enunciado](proposal.md). A página apresenta exemplos de bases e
uma regra geral de aprovação; não publica decisões individuais sobre equipes. Registramos a seleção
feita pelo assistente, sem inventar comprovante de envio de formulário ou aprovação individual.
A mesma base será mantida na futura entrega de classificação.

## 1. Initial inspection

### A. Data dictionary

Cada linha descreve um registro individual, com características e indicador de AVC. A fonte não
identifica a instituição ou o desenho de coleta. O download é o ZIP oficial do Kaggle, fixado na
versão 1; o CSV é conferido pelo SHA-256 `{SHA256}` antes da leitura.

**Tabela 1 — Dicionário de todas as 12 colunas.** Flags 0/1 são categorias; `id` não é um preditor.

{md(dictionary)}

A unidade de glicose não está explicitada na descrição da fonte; preservamos a escala e não aplicamos
limiares clínicos. Idade fracionária é mantida, inclusive abaixo de um ano. O campo `bmi` é interpretado
como IMC; não há dados de peso e altura para recalculá-lo. `Govt_job` é o código efetivamente presente
no CSV. `Unknown` em tabagismo não significa nunca ter fumado.

**Atribuição e uso:** fedesoriano, *Stroke Prediction Dataset*, Kaggle (2021), versão 1. Os metadados
atribuem os direitos dos arquivos aos autores originais e indicam uso educacional, com fonte
confidencial. Não atribuímos CC0 ou CC BY à base. O CSV bruto não é redistribuído pelo repositório;
a reprodução faz download diretamente da fonte.

### B. Quality

**Tabela 2 — Ausência e cardinalidade antes do tratamento.** `ausentes` conta NaN de armazenamento;
`ausencia_declarada` inclui também `Unknown`, descrito como informação indisponível pela fonte.

{md(tables["quality"], index=True)}

Há **201 valores NaN em `bmi` ({pct(raw.bmi.isna().mean() * 100)})** e
**{num(counts["unknown_smoking_rows"])} registros `Unknown` em tabagismo
({pct(raw.smoking_status.eq("Unknown").mean() * 100)})**. Assim, a maior ausência de informação
é tabagismo, embora o maior número de NaN seja em IMC. Não confundimos essas duas definições.
Há **{counts["duplicates_removed"]} duplicatas excedentes** considerando os 10 atributos e o alvo,
sem o identificador; restam **{num(len(clean))} linhas**. Nenhuma coluna é constante.

**Tabela 3 — Regras de integridade e domínio, definidas antes da análise de distribuições.**
Os NaN de IMC são contabilizados na Tabela 2, não como violação de positividade.

{md(tables["domain_checks"])}

Não há violação dessas regras. Isso não comprova exatidão de medição: valores muito altos de IMC
ou glicose são auditados como caudas, sem substituição por uma hipótese clínica não documentada.

**Colunas descartadas e vazamento:** `id` é removido porque identifica o registro e não descreve
um fenômeno generalizável; `stroke` é separado como alvo. Os demais campos não contêm uma cópia
explícita do alvo, mas a ausência de datas impede atestar que hipertensão, doença cardíaca e demais
informações foram coletadas antes do AVC. A tarefa atual é descrever o alvo registrado. Para uma
aplicação prospectiva será necessário obter a cronologia e auditar esse risco.

### C. Target

**Tabela 4 — Distribuição das duas classes em cada partição.**

{md(tables["classes"])}

São **{num(raw[TARGET].sum())} registros positivos ({pct(raw[TARGET].mean() * 100)})** e
**{num(raw[TARGET].eq(0).sum())} negativos ({pct(raw[TARGET].eq(0).mean() * 100)})**;
a razão é **{raw[TARGET].eq(0).sum() / raw[TARGET].sum():.2f} negativos por positivo**.
Uma regra constante de classe 0 teria acurácia de **{pct(raw[TARGET].eq(0).mean() * 100)}**,
sem identificar nenhum positivo. Esse número deriva das frequências, sem treinar um modelo.

![Figura 1 — Distribuição do alvo](figures/01_target.png)

**Conclusão da Figura 1:** o forte desbalanceamento torna acurácia isolada pouco informativa;
na modelagem futura serão necessários PR-AUC, recall, precisão e matriz de confusão.

### D. Train and test

A regra de exclusão de `id` e a deduplicação são determinísticas e não estimam parâmetros.
Depois delas, antes de decisões baseadas em distribuições, fixamos a primeira divisão de
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. O grupo é o hash dos 10 atributos,
sem `id` nem alvo: perfis iguais, mesmo com alvos divergentes, permanecem juntos.
Há **{num(counts["profile_groups"])} perfis distintos** e **{counts["repeated_profile_rows"]} linhas
além do primeiro registro por perfil**. Uma asserção confirma zero perfis compartilhados.

O treino contém **{num(len(train))} linhas ({pct(len(train) / len(clean) * 100)})** e o teste,
**{num(len(test))} ({pct(len(test) / len(clean) * 100)})**. A frequência positiva é
**{pct(train[TARGET].mean() * 100)} no treino** e **{pct(test[TARGET].mean() * 100)} no teste**:
a estratificação por grupos é aproximada. A [lista de linhas e partições](tables/split.csv)
permite auditoria. `source_row` é índice de rastreamento (zero-based, sem cabeçalho), nunca atributo.

Não há datas para uma divisão temporal real, nem identificador de pessoa validado que permita
verificar reincidência. Essa divisão não demonstra desempenho em períodos futuros. As seções 2–4
usam exclusivamente o treino; o teste recebe apenas a transformação ajustada e checagens de contrato.

## 2. Univariate analysis

### A. Numerical

**Tabela 5 — Estatísticas dos três atributos numéricos no treino, antes da imputação.**
Média, mediana, desvio amostral, quartis e P99 são calculados somente entre valores observados;
`n observado` explicita o denominador de cada coluna. Tabelas usam ponto decimal.

{md(description, index=True)}

![Figura 2 — Distribuições numéricas](figures/02_numeric.png)

**Conclusão da Figura 2:** idade ocupa uma faixa ampla e tem assimetria pequena; glicose e IMC
apresentam caudas à direita. São medidas de formas diferentes, não três variáveis normais equivalentes.

{chr(10).join(numerical_notes)}

Os painéis usam 30 bins na escala original para expor forma e extremos sem ocultar a unidade.
Os picos dependem dos bins; não inferimos um número exato de modas nem subpopulações clínicas a
partir do histograma. Os valores de IMC ausentes não entram no histograma. A auditoria IQR é
quantificada na seção 4A, e não remove automaticamente valores altos.

### B. Categorical

**Tabela 6 — Cardinalidade e número de níveis raros (<1% do treino).**

{md(tables["cardinality"])}

**Tabela 7 — Frequências completas de todas as sete variáveis categóricas.**

{md(cats)}

![Figura 3 — Frequências categóricas](figures/03_categorical.png)

**Conclusão da Figura 3:** os níveis têm suportes distintos e há categorias raras;
nenhum atributo tem alta cardinalidade (máximo **{int(tables["cardinality"].cardinalidade.max())} níveis**).
One-hot é viável sem impor ordem aos códigos ou agregar grupos pequenos arbitrariamente.

{chr(10).join(categorical_notes)}

`gender=Other` tem suporte insuficiente para concluir algo sobre esse grupo. `Unknown` em tabagismo
é mantido como categoria explícita: imputá-lo como nunca fumou mudaria o significado do dado.

## 3. Bivariate and multivariate analysis

### A. Numerical × numerical

Usamos **Spearman**, porque glicose e IMC têm caudas e a associação com idade pode ser monotônica,
sem ser linear. Empates recebem ranks médios. A correlação usa pares completos observados,
sem preencher IMC com a mediana para criar artificialmente repetições.
A [matriz completa em CSV](tables/correlation.csv) permite conferir os valores sem arredondamento visual.

![Figura 4 — Correlação de Spearman](figures/04_correlation.png)

**Conclusão da Figura 4:** o maior par é **{pair.variavel_1} × {pair.variavel_2}**, com
**ρ={pair.rho:.6f}**, entre **{num(pair.n_pares_completos)} pares completos**. A associação é moderada;
nenhum par indica redundância quase perfeita que justifique descartar um dos três atributos.

**Tabela 8 — Todos os três pares, ordenados pelo valor absoluto de ρ.**

{md(tables["correlation_pairs"], precision=6)}

![Figura 5 — Dispersão de idade, IMC e glicose](figures/05_scatter.png)

**Conclusão da Figura 5:** há dispersão dentro de cada idade e sobreposição das classes; as relações
não são cópias determinísticas. A nuvem de IMC não inclui os registros ausentes. Idade e glicose
têm ρ={tables["correlation"].loc["age", "avg_glucose_level"]:.6f}, menor que o maior par da Tabela 8.
Preservamos os três atributos e avaliamos suas contribuições apenas na modelagem posterior.

### B. Categorical × target

**Tabela 9 — Frequência de AVC registrado por categoria, com denominadores e intervalo Wilson 95%.**
São estimativas descritivas por grupo, sem ajuste por idade, confundimento ou comparações múltiplas;
não são probabilidades populacionais nem inferências causais.

{md(conditional)}

![Figura 6 — Alvo por categoria](figures/06_category_target.png)

**Conclusão da Figura 6:** as frequências variam entre grupos, mas categorias pequenas têm intervalos
amplos; 0 positivos em um grupo pequeno não demonstra risco nulo. Entre categorias com n≥100:

{chr(10).join(relation_notes)}

Diferenças em `ever_married` e `work_type` podem refletir a composição etária; não interpretamos
casamento ou trabalho como causa de AVC. O mesmo cuidado vale para comorbidades e tabagismo.

### C. Numerical × categorical

Os boxplots confrontam idade e glicose com hipertensão, separando também o alvo. Caixas mostram
Q1, mediana e Q3; whiskers usam 1,5 IQR. Pontos externos são ocultados apenas no desenho,
sem excluir registros ou limitar valores na análise.

![Figura 7 — Boxplots por hipertensão e alvo](figures/07_grouped.png)

**Tabela 10 — Localização e dispersão por hipertensão, agregando as classes.**

{md(grouped)}

**Conclusão da Figura 7:** a diferença entre grupos envolve localização e dispersão. No grupo com
hipertensão, idade tem mediana maior e IQR menor; glicose tem mediana maior e IQR maior. Logo,
comparar apenas médias ou assumir dispersão igual esconderia estrutura observada no treino.

{chr(10).join(box_notes)}

## 4. Preprocessing

### A. Strategies

**Tabela 11 — Ausência de IMC e tabagismo no treino, por classe.**

{md(tables["missingness_target"])}

A ausência não tem a mesma frequência nas duas classes; isso não permite afirmar que seja aleatória.
Não descartamos linhas com IMC ausente, pois isso alteraria a composição do alvo.

| Decisão | Evidência e implementação |
| --- | --- |
| Ausentes numéricos | Há 201 NaN no bruto (Tabela 2) e **{counts["train_bmi_missing"]} no treino**. A mediana de IMC aprendida só no treino é **{counts["train_bmi_median"]:.3f}**; ela preenche IMC antes de log1p. O indicador `missing__missingindicator_bmi` preserva a ausência em 0/1 (Tabela 11). |
| Ausentes categóricos | `Unknown` em tabagismo é informação indisponível e permanece como nível próprio (Tabelas 2 e 7). Para um NaN futuro em categóricos, imputamos o texto `Ausente`, sem trocar por uma categoria conhecida. |
| Outliers | Glicose e IMC têm caudas (Tabela 5, Figura 2). Usamos `log1p` nessas duas colunas e preservamos todos os registros; limites IQR abaixo são diagnósticos. Idade permanece sem log. |
| Codificação | As sete categorias têm no máximo {int(tables["cardinality"].cardinalidade.max())} níveis (Tabela 6). One-hot mantém todos os níveis conhecidos; `handle_unknown='ignore'` transforma nível inédito em zeros naquele bloco. |
| Escala | Idade, glicose e IMC têm unidades e dispersões diferentes (Tabela 5). StandardScaler usa médias/desvios do treino, após log1p nas duas colunas de cauda e diretamente na idade. |
| Descarte | `id` é identificador (Tabela 1); `stroke` fica exclusivamente como alvo. |

**Tabela 12 — Auditoria de outliers no treino: fora de [Q1−1,5×IQR, Q3+1,5×IQR].**

{md(tables["outliers"], index=True)}

Há **{num(counts["outlier_train_rows"])} linhas de treino** fora de pelo menos um limite.
Foram **0 linhas removidas e 0 limitadas** por esse diagnóstico. Log1p altera ao menos uma entrada
em **{num(counts["log_changed_train_rows"])} linhas**, inclusive as que não ultrapassam IQR.
Isso comprime caudas continuamente, sem declarar que um extremo válido é erro.

Indicadores one-hot e o indicador de IMC ausente ficam em 0/1, sem padronização para não amplificar
níveis raros. As três medidas numéricas ficam com variância unitária após transformação. Essa escolha
de ponderação dá mais peso às medidas contínuas na distância euclidiana; deve ser considerada na
leitura das projeções e revisada em validação futura, sem consultar o teste.

![Figura 8 — Glicose antes e depois da transformação](figures/08_transformation.png)

**Conclusão da Figura 8:** log1p comprime a cauda da glicose; a padronização posterior
centraliza e muda a escala, sem mudar novamente a forma. Os dois painéis mostram os mesmos
registros, com bins nas respectivas unidades. Um histograma mais simétrico não comprova melhora
preditiva: log1p é uma escolha inicial a comparar com a escala original em validação interna
do treino, como enfatiza a aula de transformações. Nenhum modelo foi avaliado nesta entrega.

### B. Dimensionality reduction

PCA recebe a matriz preparada de **{num(len(train))} registros de treino**, sem alvo nem `id`.
Para comparar mapas fixamos a mesma amostra estratificada de **2.000 registros de treino** com seed 42.
A estratificação serve à representatividade visual; o alvo somente colore pontos e nunca é passado
ao ajuste de PCA, t-SNE ou UMAP.

Parâmetros: PCA com SVD completa; t-SNE com perplexity **15 e 50**, inicialização PCA,
learning_rate automático, 1.000 iterações; UMAP com n_neighbors **15 e 50**, min_dist 0.1,
distância euclidiana. Todos os random_state e transform_seed explícitos são **42**, com uma thread.

![Figura 9 — Variância explicada da PCA](figures/09_pca_variance.png)

**Conclusão da Figura 9:** PC1 explica **{pct(pca.explained_variance_ratio_[0] * 100)}**, PC2
**{pct(pca.explained_variance_ratio_[1] * 100)}**; juntas, **{pct(cumulative[1] * 100)}**.
São necessárias **{pc90} componentes** para pelo menos 90%. O mapa 2D deixa informação de fora.

**Tabela 13 — Cinco maiores pesos absolutos por componente.** Aqui loadings significa pesos de
`components_`, não correlações variável-componente. O sinal global de uma PC pode inverter sem mudar
a geometria. [Todos os pesos](tables/pca_weights.csv).

{md(top_weights, precision=6)}

{chr(10).join(component_notes)}

Esses pesos descrevem combinações de idade, medidas transformadas e categorias, não causas do
alvo nem importância preditiva. O indicador de ausência é lido como característica de registro.

![Figura 10 — PCA, t-SNE e UMAP na mesma amostra](figures/10_projections.png)

**Conclusão da Figura 10:** mapas não lineares revelam concentrações locais que o mapa linear
sobrepõe, mas a distribuição visual muda com os parâmetros e as classes continuam misturadas.
Tamanho das ilhas e distância entre ilhas em t-SNE/UMAP não medem prevalência nem dissimilaridade global.
Não usamos uma ilha como diagnóstico, classe descoberta ou prova de desempenho de modelo.

**Tabela 14 — Trustworthiness em 10 vizinhos, na mesma amostra e no espaço preparado.**
Valores próximos de 1 indicam menos vizinhos espúrios introduzidos pelo mapa, sem medir separação de
classes ou generalização. [Coordenadas e índices dos cinco mapas](tables/projections.csv).

{md(diagnostics, precision=6)}

PCA obtém **{diagnostics.iloc[0].trustworthiness:.6f}**; t-SNE (15/50),
**{diagnostics.iloc[1].trustworthiness:.6f} / {diagnostics.iloc[2].trustworthiness:.6f}**; UMAP (15/50),
**{diagnostics.iloc[3].trustworthiness:.6f} / {diagnostics.iloc[4].trustworthiness:.6f}**.
A comparação quantifica preservação local; não seleciona um preditor. Mesmo uma projeção com
vizinhanças bem preservadas pode manter forte sobreposição do alvo.

### C. Pipeline

`code/preprocessing.py` é importável e retorna uma `Pipeline` com `ColumnTransformer`.
`analyze.py` ajusta uma única vez por `fit_transform(X_train)` e chama somente `transform(X_test)`.
Imputação, médias, desvios e vocabulários categóricos são aprendidos exclusivamente no treino.

**Tabela 15 — Contrato executado e verificado.**

| Verificação | Resultado |
| --- | --- |
| Treino antes/depois | {len(train)} × {len(FEATURES)} → {train_matrix.shape} |
| Teste antes/depois | {len(test)} × {len(FEATURES)} → {test_matrix.shape} |
| NaN ou infinitos nas saídas | 0 em ambas |
| Perfis compartilhados entre partições | 0 |
| Categoria real inédita no teste | {checks["unseen_test_categories"]} |
| Categoria sintética Nunca_observado em work_type | saída finita; mesmo shape; bloco work_type todo zero |
| NaN sintético em bmi e gender | saída finita, com imputação e indicador de IMC |

Como nenhum nível de referência foi descartado, um bloco todo zero para uma categoria inédita não
se confunde com uma categoria conhecida de referência. A identidade da categoria nova é perdida;
é necessário monitorar níveis inéditos na entrada. O indicador de ausência de IMC tem dimensão fixa,
inclusive se um fold não tiver IMC ausente. Os casos sintéticos não modificam a base real.

**Tabela 16 — Nomes e ordem de todas as {len(feature_names)} colunas finais.**

{md(names_table)}

**Tabela 17 — Faixas e momentos finais das três medidas numéricas.**

{md(tables["scaling_checks"], precision=6)}

StandardScaler usa desvio populacional (ddof=0): a média é aproximadamente zero e o desvio
é um no treino; isso não é uma exigência para o teste. As médias/desvios descritivos da Tabela 5
usam dados observados e desvio amostral, portanto não devem ser confundidos com estes valores.
A padronização não limita a saída a [-1,1]. O exercício 3 aceita StandardScaler ou MinMaxScaler;
esta EDA usa a primeira alternativa. Não afirmamos que a faixa seja limitada nem que tanh esteja
livre de saturação; avaliar a ativação e uma alternativa MinMax em validação pertence à modelagem.
One-hot e indicador de ausência continuam em 0/1; o [CSV completo](tables/scaling_checks.csv)
permite conferir as faixas numéricas sem inferir parâmetros do teste.

Os testes comparam medianas, médias e vocabulários com o treino, transformam um teste sintético com
valores extremos e verificam que o estado ajustado não muda. Também verificam o descarte de `id` e
alvo, grupos, integridade do CSV, indicador de ausência e links dos arquivos no site compilado.
[Como reproduzir](reproduce.md).

```python
--8<-- "docs/projects/eda/code/preprocessing.py"
```

## 5. Synthesis

Para quem preparará o modelo:

- **Desbalanceamento:** só {pct(clean[TARGET].mean() * 100)} dos registros são positivos
  (Tabela 4, Figura 1). Usar métricas adequadas e validação estratificada dentro do treino; pesos,
  reamostragem e escolha de limiar pertencem aos folds de treino/validação, sem consultar o teste.
- **Ausência informativa:** IMC tem 201 ausentes no bruto e tabagismo tem
  {num(counts["unknown_smoking_rows"])} Unknown (Tabela 2). As frequências variam por alvo no treino
  (Tabela 11). Mediana preserva o tamanho da matriz; o indicador e Unknown sinalizam ausência, sem recuperar o valor real;
  monitorar esses padrões em novos dados e revisar sua origem.
- **Caudas e escalas:** Tabela 5 e Figura 2 justificam log1p em glicose/IMC e padronização.
  {num(counts["outlier_train_rows"])} registros ultrapassam limites IQR (Tabela 12), mas não foram
  excluídos sem evidência de erro. Revisar extremos com a fonte, sem impor limiares clínicos inventados.
- **Associações e confundimento:** maior correlação numérica é {pair.rho:.6f} (Tabela 8, Figuras 4–5).
  Tabelas 9–10 e Figuras 6–7 mostram diferenças por grupos que podem refletir idade e composição
  amostral. Não transformar associação em causalidade ou usar categoria rara como regra conclusiva.
- **Redução:** PC1+PC2 preserva {pct(cumulative[1] * 100)} da variância (Figura 9).
  A Figura 10 e Tabela 14 comparam preservação local e sensibilidade dos mapas, sem evidência de
  separação perfeita. O pipeline final preserva todas as colunas preparadas; redução não é obrigatória.
- **Origem e temporalidade:** a fonte é confidencial e não há datas. Antes de qualquer alegação de
  previsão futura, obter cronologia, representatividade, critérios de rótulo e disponibilidade prévia
  dos atributos. A EDA atual se limita a registros observados para finalidade educacional.

A base foi escolhida dentre os exemplos explicitamente indicados no enunciado, por seleção do assistente; a confirmação específica pelo aluno permanece em aberto. Código, relatório, figuras e tabelas são versionados; dados brutos são obtidos da fonte
com versão e hash fixados. O README e o front matter declaram integralmente o uso de IA.

### Results summary

{md(summary)}
"""
    (output / "index.md").write_text(text, encoding="utf-8")
