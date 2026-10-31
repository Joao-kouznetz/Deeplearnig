# Reproduzir a análise de Stroke Prediction

Execute os comandos na raiz do repositório. Python 3.12 é o ambiente suportado;
as versões das dependências científicas e do site estão fixadas em
`requirements.txt`. A execução foi validada em Linux com Python 3.12.16.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python docs/projects/eda/code/analyze.py
python -m pytest -q
ruff check docs/projects/eda/code tests
ruff format --check docs/projects/eda/code tests
mkdocs build --strict
```

O download inicial requer internet. Depois, o CSV é lido localmente, sempre com
conferência de SHA-256. Se o Kaggle estiver temporariamente indisponível, execute
novamente; o programa não troca de fonte nem aceita outro arquivo silenciosamente.

## O que é refeito

1. Obtenção do ZIP oficial do Kaggle (versão 1) e verificação do CSV.
2. Remoção determinística de duplicatas completas e divisão por perfis, seed 42.
3. Inspeção inicial e estatísticas/figuras das seções 2–3 somente no treino.
4. Ajuste do pré-processamento no treino, transformação do teste e verificação de
   ausentes, infinitos e categorias desconhecidas; mediana e indicador para IMC ausente.
5. PCA no treino completo e cinco mapas na mesma amostra de 2.000 registros de treino:
   PCA, t-SNE (perplexity 15/50), UMAP (n_neighbors 15/50).
6. Escrita de `index.md`, nove PNGs numerados e tabelas em CSV/JSON.

PCA e embeddings são métodos de exploração, não classificadores. O alvo nunca é
passado ao ajuste dos redutores. Todas as seeds explícitas são 42, e o processo
limita as bibliotecas a uma thread. UMAP/Numba pode compilar código na primeira
execução. Bibliotecas, arquitetura e BLAS diferentes podem causar pequenas
diferenças numéricas ou na posição de embeddings, mesmo com a mesma seed.

## Usar o pipeline em outro arquivo

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("docs/projects/eda/code").resolve()))
from dataset import FEATURES, download, load_data, split_data
from preprocessing import make_preprocessor

download()
_, train, test = split_data(load_data())
preprocess = make_preprocessor()
X_train_t = preprocess.fit_transform(train[FEATURES])
X_test_t = preprocess.transform(test[FEATURES])
feature_names = preprocess.get_feature_names_out()
```

A matriz densa contém 24 colunas no split publicado. `make_preprocessor()` cria
um objeto novo, sem ajustar estatísticas; em validação cruzada futura, construa
um pipeline completo novo em cada fold. Os códigos categóricos devem ser strings
como na saída de `load_data()`. Os atributos numéricos precisam respeitar o
domínio auditado no relatório; IMC pode ser ausente. `id` e `stroke` não são entradas.
O indicador de ausência de IMC permanece em 0/1, junto com os indicadores one-hot.

O comando principal também grava `artifacts/preprocessor.joblib` para uso local,
fora do Git. A reprodução é definida pelo código e pelas dependências, não pela
distribuição de um binário ajustado.

## Arquivos de auditoria

| Arquivo em `tables/` | Conteúdo |
| --- | --- |
| `quality.csv`, `domain_checks.csv` | Ausentes, cardinalidade e regras de domínio |
| `classes.csv`, `split.csv` | Distribuições do alvo e rastreamento de cada linha original |
| `numeric.csv`, `categorical.csv`, `cardinality.csv` | Estatísticas completas do treino |
| `correlation.csv`, `correlation_pairs.csv` | Spearman e ranking dos pares |
| `category_target.csv`, `grouped_numeric.csv` | Taxas por categoria e quartis por hipertensão |
| `missingness_target.csv` | Ausências de IMC e tabagismo por alvo no treino |
| `outliers.csv` | Limites IQR calculados no treino; nenhuma remoção por esses limites |
| `feature_names.csv` | Ordem e nomes das colunas transformadas |
| `pca_weights.csv`, `pca_top_weights.csv` | Pesos completos e maiores pesos das duas primeiras PCs |
| `projections.csv`, `projection_diagnostics.csv` | Índices, coordenadas e preservação de vizinhanças |
| `metrics.json`, `results_summary.csv` | Números principais e tabela final da rubrica |

Os testes usam os dados reais já baixados e casos sintéticos de contrato. Para
rodar apenas os testes do pipeline, sem dados externos, use
`python -m pytest tests/test_preprocessing.py -q`.

## Ler o site

`mkdocs build --strict` gera HTML em `site/`; o arquivo inicial é `site/index.html`.
No GitHub, o workflow de validação publica um artefato `site-eda` com o site
compilado, acessível na execução do Actions. A publicação em GitHub Pages é
acionada por push na `main` ou manualmente pelo workflow após ele existir na
branch padrão. A fonte de Pages deve ser **GitHub Actions** nas configurações do
repositório. Não é necessário servidor para construir ou revisar os arquivos.

[Voltar ao relatório](index.md).
