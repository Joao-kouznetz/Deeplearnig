"""Escreve o relatório usando os mesmos resultados que originaram os CSVs e figuras."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataset import (
    ALL_NUMERIC,
    CATEGORICAL,
    ENGAGEMENT,
    FEATURES,
    PROJECT,
    SHA256,
    SOURCE,
    TARGET,
)

DICTIONARY = [
    ("Administrative", "Páginas administrativas visitadas", "Numérica discreta", "páginas"),
    (
        "Administrative_Duration",
        "Tempo acumulado em páginas administrativas",
        "Numérica contínua",
        "tempo¹",
    ),
    ("Informational", "Páginas informativas visitadas", "Numérica discreta", "páginas"),
    (
        "Informational_Duration",
        "Tempo acumulado em páginas informativas",
        "Numérica contínua",
        "tempo¹",
    ),
    ("ProductRelated", "Páginas de produtos visitadas", "Numérica discreta", "páginas"),
    (
        "ProductRelated_Duration",
        "Tempo acumulado em páginas de produtos",
        "Numérica contínua",
        "tempo¹",
    ),
    (
        "BounceRates",
        "Taxa de rejeição das páginas visitadas, segundo Analytics",
        "Numérica contínua",
        "fração [0, 1]",
    ),
    (
        "ExitRates",
        "Taxa de saída das páginas visitadas, segundo Analytics",
        "Numérica contínua",
        "fração [0, 1]",
    ),
    (
        "PageValues",
        "Valor médio atribuído às páginas antes de transações; excluído do pipeline",
        "Numérica contínua",
        "valor; moeda não informada",
    ),
    (
        "SpecialDay",
        "Proximidade de uma data comercial especial",
        "Numérica ordinal",
        "índice [0, 1]",
    ),
    ("Month", "Mês da sessão; códigos originais, inclusive June", "Categórica nominal", "mês"),
    (
        "OperatingSystems",
        "Código de sistema operacional, sem ordem semântica",
        "Categórica nominal",
        "código",
    ),
    ("Browser", "Código do navegador, sem ordem semântica", "Categórica nominal", "código"),
    (
        "Region",
        "Código da região, sem identificação geográfica no arquivo",
        "Categórica nominal",
        "código",
    ),
    ("TrafficType", "Código de origem de tráfego", "Categórica nominal", "código"),
    ("VisitorType", "Visitante novo, recorrente ou Other", "Categórica nominal", "categoria"),
    ("Weekend", "Se a sessão ocorreu no fim de semana", "Categórica binária", "True / False"),
    ("Revenue", "Se a sessão resultou em compra; alvo", "Categórica binária", "0 = não; 1 = sim"),
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
    numeric = tables["numeric"]
    cats = tables["categorical"]
    conditional = tables["category_target"]
    top_pair = tables["correlation_pairs"].iloc[0]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pc90 = int(np.searchsorted(cumulative, 0.9) + 1)
    minority = clean[TARGET].mean() * 100
    dictionary = pd.DataFrame(
        DICTIONARY, columns=["Variável", "Significado", "Tipo semântico", "Unidade"]
    )
    description = numeric[
        ["count", "mean", "50%", "std", "min", "25%", "75%", "99%", "max", "skewness", "zero_pct"]
    ].copy()
    description.columns = [
        "n",
        "média",
        "mediana",
        "desvio",
        "mín",
        "Q1",
        "Q3",
        "P99",
        "máx",
        "assimetria",
        "zeros (%)",
    ]
    numerical_notes = []
    for col in ALL_NUMERIC:
        row = numeric.loc[col]
        kind = "cauda à direita" if row.skewness > 1 else "assimetria moderada ou pequena"
        numerical_notes.append(
            f"- **{col}:** mediana {row['50%']:.3f}, média {row['mean']:.3f}, "
            f"assimetria {row.skewness:.3f} e {pct(row.zero_pct)} de zeros: {kind}. "
            + (
                "O pico em zero representa ausência de navegação nesse tipo de página."
                if col in ENGAGEMENT
                else "Não interpretamos a distribuição como normal."
            )
        )
    categorical_notes = []
    for col in CATEGORICAL:
        tab = cats[cats.variavel.eq(col)]
        majority = tab.iloc[0]
        rare = tab.loc[tab.rara_menos_1pct, "categoria"].astype(str).tolist()
        categorical_notes.append(
            f"- **{col}:** {len(tab)} categorias; a mais frequente é `{majority.categoria}` "
            f"({num(majority.n)} sessões; {pct(majority.pct)}). "
            f"Categorias com frequência <1%: {', '.join(f'`{c}`' for c in rare) or 'nenhuma'}."
        )
    relation_notes = []
    for col in CATEGORICAL:
        tab = conditional[conditional.variavel.eq(col)]
        eligible = tab[tab.n.ge(100)]
        low, high = (
            eligible.loc[eligible.taxa_pct.idxmin()],
            eligible.loc[eligible.taxa_pct.idxmax()],
        )
        relation_notes.append(
            f"- **{col}:** entre categorias com ao menos 100 sessões, a taxa varia de "
            f"{pct(low.taxa_pct)} em `{low.categoria}` (n={num(low.n)}) a "
            f"{pct(high.taxa_pct)} em `{high.categoria}` (n={num(high.n)}). "
            "São associações observacionais, sem ajuste por outras variáveis."
        )
    weights = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=["PC1", "PC2"])
    weights.index.name = "feature"
    weights.to_csv(output / "tables/pca_weights.csv", float_format="%.10g")
    loadings = []
    for component in ["PC1", "PC2"]:
        for name in weights[component].abs().nlargest(5).index:
            loadings.append(
                {"componente": component, "feature": name, "peso": weights.loc[name, component]}
            )
    loading_table = pd.DataFrame(loadings)
    loading_table.to_csv(output / "tables/pca_top_weights.csv", index=False, float_format="%.10g")
    main_pc1 = weights.PC1.abs().nlargest(3).index.tolist()
    main_pc2 = weights.PC2.abs().nlargest(3).index.tolist()
    strongest = tables["correlation_pairs"].head(6)
    names_table = pd.DataFrame(
        {"posição (0-based)": range(len(feature_names)), "feature": feature_names}
    )
    grouped = tables["grouped_numeric"]
    grouped_notes = []
    for col in ["ProductRelated", "ExitRates"]:
        rows = grouped[grouped.variavel.eq(col)]
        values = "; ".join(
            f"{r.VisitorType}: mediana={r.mediana:.3f}, IQR={r.IQR:.3f}" for r in rows.itertuples()
        )
        grouped_notes.append(f"- **{col}**, agregando os alvos: {values}.")
    nonzero_issues = tables["domain_checks"].query("n > 0")
    issue_text = (
        "Todas as regras da Tabela 3 resultam em zero violações. Isso não comprova ausência de erro de medição."
        if nonzero_issues.empty
        else "A Tabela 3 quantifica as violações. As combinações de duração e contagem não são corrigidas "
        "por inferência: o arquivo não permite reconstruir a sessão; ficam documentadas como limitação."
    )
    summary = pd.DataFrame(
        [
            [
                1,
                "Dataset, tarefa e alvo",
                "Online Shoppers Purchasing Intention (UCI 468); classificação; Revenue",
            ],
            [
                2,
                "Instâncias × atributos (numéricos / categóricos)",
                f"{num(len(raw))} × 17 (10 / 7), mais o alvo; {num(len(clean))} × 16 (9 / 7) após regras",
            ],
            [3, "Coluna com mais ausentes e percentual", "Empate entre todas: 0 (0%)"],
            [
                4,
                "Colunas descartadas e motivo",
                "PageValues: risco de vazamento; Revenue separado como alvo",
            ],
            [
                5,
                "Classe minoritária (%)",
                f"Compra=1: {pct(minority)} na base sem duplicatas; {pct(raw[TARGET].mean() * 100)} no bruto",
            ],
            [
                6,
                "Tamanhos de treino e teste",
                f"{num(len(train))} / {num(len(test))} (seed=42; grupos + estratificação)",
            ],
            [
                7,
                "Par numérico mais correlacionado e valor",
                f"{top_pair.variavel_1} × {top_pair.variavel_2}; Spearman ρ={top_pair.rho:.6f}",
            ],
            [
                8,
                "Linhas afetadas pela estratégia de outliers",
                f"Treino: {num(counts['outlier_train_rows'])} sinalizadas pelo IQR; 0 removidas/limitadas; {num(counts['log_changed_train_rows'])} alteradas por log1p",
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
    text = f"""---
project: eda
task: classification
dataset: {SOURCE}
team:
  - João K Bresser
ai_use: "Projeto desenvolvido com inteligência artificial (OpenAI Codex): seleção da base, código, análise, gráficos, testes e redação. Resultados calculados executando o código; revisão humana e aprovação da base pelo professor ainda necessárias."
---

# Projeto 1 — Análise exploratória de intenção de compra

Este projeto investiga **quais características de navegação se associam à compra em uma sessão**.
O alvo discreto é `Revenue` (0/1). O produto desta entrega é um pipeline para uma futura rede neural;
nenhum classificador foi treinado e não há estimativa de desempenho preditivo.

**Escopo de uso:** classificação retrospectiva de sessões completas. O CSV não contém eventos
intermediários nem o instante da compra, portanto esta análise não demonstra previsão em tempo real.
As seções 2–4 usam **somente o treino**; o teste é usado para conferir o contrato de transformação.

**Transparência:** este trabalho foi feito usando inteligência artificial, inclusive a redação das
interpretações. Cada número abaixo é calculado pelo código versionado, sem resultados inventados.
É necessário estudar e revisar as decisões antes de defender o trabalho.

**Referências:** [enunciado e rubrica](https://insper.github.io/ann-dl/2026.2/projects/eda/),
[formato de entrega](https://insper.github.io/ann-dl/2026.2/exercises/submission/),
[fonte primária UCI]({SOURCE}), [DOI](https://doi.org/10.24432/C5F88Q),
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
[parâmetros do UMAP](https://umap-learn.readthedocs.io/en/latest/parameters.html).
Tabelas e gráficos foram produzidos por esta implementação, sem reutilizar análises de terceiros.

## 0. Proposal

A proposta é usar **Online Shoppers Purchasing Intention**, de C. Sakar e Yomi Kastro (2018),
licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
O arquivo tem **{num(len(raw))} sessões × 18 colunas**: 10 atributos numéricos,
7 categóricos e o alvo binário. É público, tabular, tem mais de 1.000 linhas e mais de 5 atributos.
Não é uma das bases clássicas expressamente vedadas no enunciado.

O problema combina comportamento de navegação, calendário e contexto do visitante, permitindo
estudar caudas longas, desbalanceamento e relações não lineares. O primeiro risco é que `PageValues`
envolve atribuição de valor a transações: sem sua data de cálculo, não é seguro usá-lo como preditor.

**Status acadêmico: proposta preparada; aprovação do professor não comprovada.** A disciplina exige
aprovação da base como pré-requisito; esta entrega técnica não substitui essa aprovação nem o registro
no formulário. [Proposta pronta para revisão e envio](proposal.md).

## 1. Initial inspection

### A. Data dictionary

Uma linha corresponde ao resumo de uma sessão. A UCI informa que a construção buscou sessões de
usuários distintos em um período de um ano; o CSV não oferece IDs para verificar essa independência.
O download usa o ZIP oficial e verifica o SHA-256 `{SHA256}`.

**Tabela 1 — Dicionário das 18 colunas originais.** Tipos semânticos prevalecem sobre o dtype do CSV.

{md(dictionary)}

¹ As durações são quantidades de tempo; a página da UCI e o CSV não explicitam a unidade. Mantemos
a escala original e não afirmamos segundos. `PageValues` também não informa moeda.
`SpecialDay` tem ordem, mas intervalos comerciais não necessariamente equivalentes; tratá-lo como
índice numérico é uma hipótese simples a revisar na modelagem. Os códigos de sistema, navegador,
região e tráfego são strings nominais no pipeline. Janeiro e abril não aparecem; não criamos sessões
ou observações nesses meses. `June` é preservado como código da fonte.

### B. Quality

**Tabela 2 — Ausentes por coluna (contagem e percentual), antes de qualquer tratamento.**

{md(tables["quality"], index=True)}

Há **{counts["missing_total"]} valores ausentes**, e todas as colunas empatam em 0% de ausência.
Há **{num(counts["duplicates_removed"])} duplicatas exatas excedentes** ({pct(counts["duplicates_removed"] / len(raw) * 100)}),
removidas por regra determinística, restando **{num(len(clean))} linhas**. Isso trata repetições
como redundância de registro; sessões distintas podem ter o mesmo resumo, então a deduplicação
também muda ligeiramente a população de análise. Não há identificador ou coluna constante a descartar.

**Tabela 3 — Auditoria de domínio no arquivo original.** Taxas fora de [0,1], contagens negativas
ou fracionárias e duração positiva com contagem zero são verificadas explicitamente.

{md(tables["domain_checks"])}

{issue_text}

**Vazamento:** `Revenue` fica separado do conjunto de entrada. `PageValues` é excluído preventivamente,
porque a definição da UCI envolve transações e não há metadados temporais que atestem cálculo anterior
ao desfecho. Não afirmamos que todo valor seja vazamento comprovado. `BounceRates` e `ExitRates`
também são agregados de Analytics com janela desconhecida: permanecem para análise retrospectiva,
mas uma aplicação futura precisará recomputá-los com dados históricos anteriores ao desfecho.
`source_row`, criado apenas para rastrear a divisão, nunca é atributo.

### C. Target

**Tabela 4 — Contagem e percentual das classes em cada população.**

{md(tables["classes"])}

No bruto há **{num(raw[TARGET].sum())} compras ({pct(raw[TARGET].mean() * 100)})** e
**{num(raw[TARGET].eq(0).sum())} não compras ({pct(raw[TARGET].eq(0).mean() * 100)})**,
razão de {raw[TARGET].eq(0).sum() / raw[TARGET].sum():.2f} não compras por compra.
Na base sem duplicatas a minoria é **{pct(minority)}**. Uma regra constante de “não compra” teria
acurácia de {pct(clean[TARGET].eq(0).mean() * 100)}, sem detectar compra alguma; esse é um cálculo
da frequência de classes, não uma avaliação de modelo.

![Figura 1 — Frequências do alvo](figures/01_target.png)

**Conclusão da Figura 1:** compras são minoritárias; na futura modelagem, usar PR-AUC,
recall e matriz de confusão, além de acurácia, com limiar escolhido em validação interna do treino.

### D. Train and test

Depois das regras determinísticas de integridade e exclusão por domínio, antes de analisar
distribuições ou estimar imputação/escala, aplicamos `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=42)` e fixamos **a primeira divisão**, sem escolher a que parece melhor.
O grupo é o hash dos 16 atributos retidos, sem `Revenue` e sem `PageValues`. Assim, vetores de
entrada iguais, inclusive com alvos diferentes, não atravessam a divisão. Há **{num(counts["profile_groups"])}
perfis distintos** e **{num(counts["repeated_profile_rows"])} linhas além do primeiro registro de cada perfil**
mesmo após retirar duplicatas completas. A separação entre esses perfis é verificada por asserção.

O treino tem **{num(len(train))} linhas ({pct(len(train) / len(clean) * 100)})**, e o teste,
**{num(len(test))} ({pct(len(test) / len(clean) * 100)})**. A estratificação por grupos é aproximada:
as taxas de compra são **{pct(train[TARGET].mean() * 100)} no treino** e
**{pct(test[TARGET].mean() * 100)} no teste**. A [lista de índices e partições](tables/split.csv)
permite auditar a separação; os índices começam em zero, excluindo o cabeçalho do CSV.

A divisão aleatória estima generalização entre sessões intercambiáveis do mesmo período. Como só
há o mês, sem ano/dia/ID, não conseguimos garantir uma avaliação temporal real. Não apresentamos
essa divisão como evidência de desempenho em meses futuros. O teste não participa das seções 2–4,
exceto na transformação final e na checagem de categorias inéditas.

## 2. Univariate analysis

### A. Numerical

**Tabela 5 — Estatísticas de todos os 10 atributos numéricos no treino.** `PageValues` aparece
apenas para auditoria, embora não seja usado pelo pipeline. O desvio é amostral; quartis e P99
seguem o cálculo padrão do pandas. Os decimais das tabelas usam ponto.

{md(description, index=True)}

Escolhemos uma única figura em painéis para não confundir quantidade de gráficos com evidência.
A escala symlog e os bins espaçados em log1p tornam visíveis a concentração próxima de zero e as
caudas; os rótulos continuam na unidade original. O treino não é alterado para desenhar a figura.

![Figura 2 — Distribuições dos atributos numéricos](figures/02_numeric.png)

**Conclusão da Figura 2:** os picos em zero e as caudas à direita tornam média e mediana diferentes;
normalidade e remoção automática de valores grandes não são boas hipóteses iniciais.
As leituras individuais, incluindo assimetria e concentração em zero, são:

{chr(10).join(numerical_notes)}

Os histogramas permitem reconhecer concentração e caudas, mas não estabelecem um número exato de
modas: ele depende dos bins. Especialmente nas durações, zeros são valores válidos e não são
convertidos em ausentes. A auditoria de extremos com limites IQR está na seção 4A.

### B. Categorical

**Tabela 6 — Cardinalidade e número de categorias raras (<1% do treino).**

{md(tables["cardinality"])}

**Tabela 7 — Frequências completas por categoria (nenhum nível omitido).**

{md(cats)}

![Figura 3 — Frequências categóricas](figures/03_categorical.png)

**Conclusão da Figura 3:** a massa é desigual entre meses, visitantes e códigos; categorias pequenas
precisam de cautela. A maior cardinalidade é **{int(tables["cardinality"].cardinalidade.max())}**,
portanto não há coluna com centenas de níveis que inviabilize one-hot neste conjunto.

{chr(10).join(categorical_notes)}

Não agregamos automaticamente categorias raras: a dimensão é pequena e a categoria pode ser
informativa. Na modelagem, avaliar regularização e estabilidade em validação, sem tomar o código
numérico como escala contínua.

## 3. Bivariate and multivariate analysis

### A. Numerical × numerical

Usamos **Spearman**, pois há assimetria, zeros, caudas longas e relações possivelmente monotônicas,
sem exigir relação linear nas unidades originais. Empates recebem ranks médios. A matriz completa
está na Figura 4 e no [CSV de correlações](tables/correlation.csv).

![Figura 4 — Matriz de Spearman](figures/04_correlation.png)

**Conclusão da Figura 4:** o maior par é **{top_pair.variavel_1} × {top_pair.variavel_2}**, com
**ρ={top_pair.rho:.6f}**. Há redundância entre medidas de volume e duração, e entre taxas,
que precisa ser considerada ao interpretar importância de variáveis.

**Tabela 8 — Seis maiores correlações em valor absoluto, excluindo diagonal e pares repetidos.**

{md(strongest, precision=6)}

![Figura 5 — Dispersões de dois pares correlacionados](figures/05_scatter.png)

**Conclusão da Figura 5:** o volume de páginas e a duração crescem juntos
(ρ={tables["correlation"].loc["ProductRelated", "ProductRelated_Duration"]:.6f}), e rejeição e saída
também se associam (ρ={tables["correlation"].loc["BounceRates", "ExitRates"]:.6f}). A dispersão
residual e a sobreposição das classes impedem tratá-los como cópias ou separadores perfeitos.
Preservamos ambos inicialmente e planejamos ablação/regularização em validação futura.

### B. Categorical × target

**Tabela 9 — Compra por todos os níveis categóricos**, com denominador, compras e intervalo Wilson
de 95%. Os intervalos descrevem incerteza binomial por grupo; não corrigem comparações múltiplas,
dependência ou confundimento. Não são testes de causalidade nem seleção automática de atributos.

{md(conditional)}

![Figura 6 — Compra por categoria e intervalos](figures/06_category_target.png)

**Conclusão da Figura 6:** há diferenças descritivas entre categorias, mas taxas extremas de grupos
pequenos vêm com intervalos amplos; não justificam uma regra comercial isolada. Comparando apenas
grupos com pelo menos 100 sessões para tornar a leitura menos instável:

{chr(10).join(relation_notes)}

### C. Numerical × categorical

Escolhemos `ProductRelated` e `ExitRates` para verificar se os tipos de visitante diferem em
volume e rejeição/saída. O boxplot mostra Q1, mediana e Q3, e whiskers de 1,5 IQR;
os pontos fora dos whiskers são ocultados **somente na figura**, sem excluir qualquer sessão.
As caixas separam também o alvo para evitar uma leitura baseada apenas em mistura de classes.

![Figura 7 — Boxplots por VisitorType e Revenue](figures/07_grouped.png)

**Tabela 10 — Mediana e dispersão por VisitorType (classes agregadas), para quantificar a leitura.**

{md(grouped)}

**Conclusão da Figura 7:** os grupos variam em localização e em dispersão; a Tabela 10 quantifica
as duas dimensões. A estratificação visual pelo alvo mostra sobreposição, então o tipo de visitante
sozinho não determina compra. `Other` tem pouco suporte, conforme Tabelas 7 e 9.

{chr(10).join(grouped_notes)}

## 4. Preprocessing

### A. Strategies

Cada regra é determinada pelo domínio ou pelo **treino**, sem escolher limites a partir do teste.

| Decisão | Evidência e execução |
| --- | --- |
| Ausentes | A Tabela 2 tem zero ausentes. Mantemos imputação por mediana aprendida no treino nos dois blocos numéricos, para entradas futuras; categóricos recebem `Ausente`. Não afirmamos ter corrigido ausentes reais. |
| Outliers | A Tabela 5 mostra caudas e zeros; a Tabela 11 quantifica extremos. Mantemos todos, pois sessões longas podem ser legítimas. `log1p` nas seis contagens/durações comprime as caudas e preserva zero; nenhum corte é estimado usando o teste. |
| Codificação | As Tabelas 6 e 7 mostram no máximo {int(tables["cardinality"].cardinalidade.max())} categorias por coluna. One-hot nos sete atributos nominais, sem ordem artificial e sem remover a primeira categoria. `handle_unknown='ignore'` mantém a dimensão para um nível inédito. |
| Escala | A Tabela 5 contrapõe contagens/durações com taxas [0,1]. `StandardScaler` após log1p nas seis variáveis de navegação e diretamente nas três taxas/índices, com médias e desvios ajustados só no treino. |
| Exclusão | `PageValues` sai por risco de vazamento descrito em 1B; `Revenue` nunca entra no ajuste. As 16 entradas restantes são 9 numéricas e 7 categóricas. |

Os indicadores one-hot permanecem em 0/1. Não os padronizamos: isso evita amplificar fortemente
categorias raras. A distância euclidiana das projeções dá variância unitária às nove variáveis
numéricas e variância menor aos indicadores; essa escolha de ponderação favorece sinais numéricos
e é uma limitação explícita da comparação, não uma propriedade natural dos dados.

**Tabela 11 — Auditoria de caudas no treino: limite superior Q3 + 1,5×IQR.**
Em variáveis com muitos zeros, o IQR pode ser zero; por isso sinalizar não significa erro.

{md(tables["outliers"], index=True)}

Há **{num(counts["outlier_train_rows"])} linhas** do treino acima de pelo menos um desses limites.
Foram **0 linhas removidas e 0 limitadas** por essa auditoria. A transformação log1p muda pelo
menos um valor em **{num(counts["log_changed_train_rows"])} linhas de treino**, incluindo sessões
não sinalizadas; ela é contínua, não um filtro binário de outliers. Os limites IQR são apenas
diagnósticos e não entram em `transform`.

### B. Dimensionality reduction

Todas as projeções recebem a mesma matriz preparada, sem `PageValues` e sem alvo. PCA é ajustado
em **{num(len(train))} sessões de treino**. Para a comparação visual e os algoritmos não lineares,
fixamos uma amostra estratificada de **2.000 sessões**, com `random_state=42`; essa estratificação
serve apenas para representar a classe minoritária na visualização. Os mesmos índices aparecem nos
cinco mapas; o alvo **apenas colore** os pontos e não entra no ajuste de PCA, t-SNE ou UMAP.

Parâmetros fixos: distância euclidiana; t-SNE com `perplexity=15` e `50`, `init='pca'`,
`learning_rate='auto'`, 1.000 iterações e seed 42; UMAP com `n_neighbors=15` e `50`,
`min_dist=0.1`, `n_jobs=1` e seeds 42. A PCA usa SVD completa. Não escolhemos a projeção que
aparenta separar melhor as classes, nem usamos qualquer embedding para treinar um classificador.

![Figura 8 — Variância explicada acumulada](figures/08_pca_variance.png)

**Conclusão da Figura 8:** PC1 explica **{pct(pca.explained_variance_ratio_[0] * 100)}**, PC2,
**{pct(pca.explained_variance_ratio_[1] * 100)}**; juntas, **{pct(cumulative[1] * 100)}**.
São necessárias **{pc90} componentes** para alcançar pelo menos 90%. O mapa 2D perde informação
e sua sobreposição não demonstra ausência de sinal nas dimensões restantes.

**Tabela 12 — Cinco maiores pesos absolutos por componente.** Chamamos de loadings os pesos
de `components_` abaixo; não são correlações variável-componente, e o sinal global de uma PC pode
inverter sem mudar sua interpretação. O [CSV completo](tables/pca_weights.csv) contém todos os pesos.

{md(loading_table, precision=6)}

PC1 é dominada por `{main_pc1[0]}`, `{main_pc1[1]}` e `{main_pc1[2]}`;
PC2, por `{main_pc2[0]}`, `{main_pc2[1]}` e `{main_pc2[2]}`. Os sinais da Tabela 12 indicam
quais entradas variam na mesma direção ou em oposição em cada eixo, na escala transformada.
Concretamente, PC1 contrapõe volume/duração de navegação (pesos positivos) à taxa de saída
(peso negativo); PC2 combina navegação informativa com rejeição/saída, em oposição à duração
em páginas de produtos. Essa leitura descreve os eixos calculados, não mecanismos causais.

![Figura 9 — PCA, t-SNE e UMAP](figures/09_projections.png)

**Conclusão da Figura 9:** comparar as concentrações locais com o mapa linear permite investigar
estrutura não linear, mas não oferece uma fronteira de decisão nem uma medida de desempenho.
As classes se sobrepõem e a disposição dos grupos muda com os parâmetros. O tamanho aparente das
ilhas e as distâncias entre ilhas em t-SNE/UMAP **não** medem prevalência nem dissimilaridade global.

**Tabela 13 — Preservação local de vizinhanças**, `trustworthiness` em 10 vizinhos no espaço
preparado, usando as mesmas 2.000 sessões. Valores mais próximos de 1 indicam menos vizinhos
espúrios introduzidos no 2D; não medem separação de classes, causalidade ou generalização.

{md(diagnostics, precision=6)}

Esses valores permitem dizer quanto cada mapa preserva vizinhanças que PCA comprime. Não provam
que uma ilha é uma classe real. A conclusão visual deve ser lida junto com a métrica e a
sensibilidade aos dois parâmetros. [Coordenadas e índices de todos os mapas](tables/projections.csv).

Na amostra publicada, PCA tem trustworthiness **{diagnostics.iloc[0].trustworthiness:.6f}**,
t-SNE com perplexity 15/50 tem **{diagnostics.iloc[1].trustworthiness:.6f} /
{diagnostics.iloc[2].trustworthiness:.6f}**, e UMAP com 15/50 vizinhos,
**{diagnostics.iloc[3].trustworthiness:.6f} / {diagnostics.iloc[4].trustworthiness:.6f}**.
Assim, os quatro mapas não lineares preservam melhor a vizinhança de 10 pontos que o mapa linear
nesta amostra. Eles exibem concentrações locais e lobos que se sobrepõem no arco da PCA, porém
compras e não compras continuam misturadas dentro dessas regiões. A mudança de formato das
ilhas entre parâmetros é evidência de sensibilidade da visualização, não mudança na população.

### C. Pipeline

O arquivo importável `code/preprocessing.py` contém `make_preprocessor()`, que retorna uma
`Pipeline` com `ColumnTransformer`. A construção não ajusta nada. `analyze.py` executa
**um único `fit_transform(X_train)`**, seguido de `transform(X_test)`; medianas, escalas e
vocabulários categóricos vêm exclusivamente do treino.

**Tabela 14 — Contrato verificado em execução.**

| Verificação | Resultado |
| --- | --- |
| Treino antes/depois | {len(train)} × {len(FEATURES)} → {train_matrix.shape} |
| Teste antes/depois | {len(test)} × {len(FEATURES)} → {test_matrix.shape} |
| NaN ou infinitos na saída | 0 em ambas as partições |
| Perfis de entrada compartilhados | 0 |
| Categoria real inédita no teste | {checks["unseen_test_categories"]} |
| Categoria sintética `Nunca_observado` em Month | transformada; mesmo shape; bloco Month todo zero |
| Ausentes sintéticos | imputados; saída finita, inclusive categoria Ausente |

Um nível inédito gera zeros apenas no bloco one-hot daquela variável; como não descartamos um nível
de referência, não há confusão com uma categoria conhecida de referência. A identidade do nível
inédito é perdida, de modo que o monitoramento futuro precisa contar esses eventos na entrada.
Valores ausentes sintéticos verificam robustez; não são adulterações da base original.

**Tabela 15 — Nomes e ordem das {len(feature_names)} colunas finais.**

{md(names_table)}

O teste automatizado compara medianas, médias e categorias com o treino, modifica radicalmente um
teste sintético e confirma que `transform` não muda o estado ajustado. Outros testes verificam a
separação por perfis, o descarte de `PageValues`/`Revenue`, categorias inéditas e integridade do CSV.
Instruções de instalação e execução estão em [Reproduzir a análise](reproduce.md).

```python
--8<-- "docs/projects/eda/code/preprocessing.py"
```

## 5. Synthesis

Para quem construirá o modelo:

- **Alvo desbalanceado:** compra é {pct(minority)} após deduplicação (Tabela 4, Figura 1).
  Usar validação estratificada por perfis dentro do treino; escolher métricas e limiar sem consultar
  o teste. Se houver reamostragem ou pesos, aplicá-los apenas nos folds de treino, nunca antes da divisão.
- **Caudas e zeros:** as estatísticas da Tabela 5 e Figura 2 justificam log1p e escala; não há evidência
  suficiente para remover toda sessão acima dos limites da Tabela 11. Monitorar mudança de escala.
- **Redundância:** o maior ρ é {top_pair.rho:.6f} (Tabela 8, Figuras 4–5). Investigar ablação e
  regularização em validação; não interpretar correlação ou pesos da PCA como causalidade.
- **Categorias e sazonalidade:** Tabelas 7 e 9 e Figuras 3 e 6 mostram suporte desigual. Monitorar
  níveis novos, revisar estimativas em grupos raros e obter mais períodos antes de alegar previsão futura.
- **Geometria:** PC1+PC2 preserva {pct(cumulative[1] * 100)} da variância (Figura 8). A Figura 9 e
  Tabela 13 mostram por que mapas não lineares precisam de parâmetros e diagnósticos, sem concluir
  que ilhas visuais são classes. O pipeline completo, sem redução obrigatória, está pronto para comparação.
- **Risco temporal e de vazamento:** `PageValues` foi excluído por precaução (1B). `BounceRates` e
  `ExitRates` ainda precisam de auditoria de disponibilidade histórica; contagens/durações resumem
  sessões completas. Para previsão durante a sessão, obter eventos e definir o instante de corte.
- **Integridade e independência:** {num(counts["duplicates_removed"])} cópias foram removidas (1B),
  e perfis iguais permanecem juntos (1D). Sem IDs não podemos verificar independência de pessoas;
  uma base futura com usuário e data deve usar divisão por pessoa e/ou tempo.

A entrega técnica inclui código executável, dados verificáveis, figuras e tabelas versionadas,
testes e site estático. **Aprovação docente da proposta e registro acadêmico seguem pendentes de
ação humana**; não foram inventados nem substituídos por execução de código.

### Results summary

{md(summary)}
"""
    (output / "index.md").write_text(text, encoding="utf-8")
