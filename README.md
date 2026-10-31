# Deeplearnig — Projeto EDA

Análise exploratória do **Stroke Prediction Dataset**, de fedesoriano no Kaggle,
para o [Projeto 1 de ANN-DL 2026.2 do Insper](https://insper.github.io/ann-dl/2026.2/projects/eda/).
Essa base está explicitamente indicada entre os exemplos de **classificação** no
enunciado. O alvo discreto é `stroke` (0/1). A entrega prepara os dados para uma
rede neural posterior; nenhum classificador é treinado nesta etapa.

**Este projeto foi feito usando inteligência artificial (OpenAI Codex)**,
incluindo implementação, gráficos, testes, análise e redação. Os resultados foram
calculados pela execução do código. O uso de IA também está declarado em `ai_use`
no relatório; o aluno precisa revisar e compreender cada decisão e número.

- [Relatório completo, na estrutura da rubrica](docs/projects/eda/index.md)
- [Proposta e vínculo da base com o enunciado](docs/projects/eda/proposal.md)
- [Instruções detalhadas de reprodução](docs/projects/eda/reproduce.md)
- [Conferência com aulas e exercícios](docs/projects/eda/course_alignment.md)
- [Pipeline importável](docs/projects/eda/code/preprocessing.py)
- [Tabelas e resultados auditáveis](docs/projects/eda/tables/)
- [Fonte, atribuição e integridade dos dados](data/README.md)

## Executar a partir de um clone limpo

Requer **Python 3.12** e internet para instalar as dependências e baixar o CSV
oficial na primeira execução. Ambiente validado com Python 3.12.16 e dependências
fixadas em `requirements.txt`.

```bash
git clone https://github.com/Joao-kouznetz/Deeplearnig.git
cd Deeplearnig
# Antes do merge, use a branch da entrega:
git switch codex/projeto-eda
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python docs/projects/eda/code/analyze.py
python -m pytest -q
ruff check docs/projects/eda/code tests
ruff format --check docs/projects/eda/code tests
mkdocs build --strict
```

Alternativa com [uv](https://docs.astral.sh/uv/) já instalado:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python docs/projects/eda/code/analyze.py
.venv/bin/python -m pytest -q
.venv/bin/mkdocs build --strict
```

O comando principal baixa a versão 1 oficial do Kaggle e confere o SHA-256, divide
os registros com seed 42, calcula as estatísticas do treino, ajusta o pipeline e
gera dez figuras, tabelas e o relatório. Não é necessária chave Kaggle para o
endpoint público utilizado. Se a fonte estiver temporariamente indisponível,
execute novamente; o programa não troca de arquivo ou fonte silenciosamente.
A primeira execução de UMAP/Numba pode levar alguns minutos.

Relatório, figuras e tabelas são arquivos de entrega versionados. CSV bruto,
ambiente, caches, site compilado e `artifacts/preprocessor.joblib` são regeneráveis
e ficam fora do Git. Após o download, a análise não precisa se conectar ao Kaggle.

## Decisões principais

- `id` é removido; o alvo `stroke` fica separado das entradas.
- Os 10 atributos úteis são 3 numéricos e 7 categóricos, com flags 0/1 tratadas como
  categorias. Perfis iguais ficam sempre na mesma partição.
- IMC tem 201 valores ausentes no arquivo original. A mediana é aprendida somente
  no treino, e um indicador 0/1 preserva a informação de ausência.
- `Unknown` em tabagismo significa informação indisponível; não é convertido em
  nunca fumou. O relatório diferencia ausência declarada de NaN no CSV.
- Log1p comprime as caudas de glicose e IMC; não há exclusão de registros por
  outliers. Escalas e categorias são ajustadas exclusivamente no treino.
- PCA, duas configurações de t-SNE e duas de UMAP usam a mesma amostra de treino.
  Ilhas visuais não são tratadas como classes ou avaliação de modelo.

A fonte não publica datas nem o desenho de coleta; o escopo é educacional e
observacional, sem alegar previsão clínica futura ou causalidade.

## Site e integração contínua

MkDocs Material gera `site/index.html` com `mkdocs build --strict`. O workflow
[Validar EDA](.github/workflows/validate.yml) reproduz a análise, roda os testes e
oferece o site como artefato `site-eda` em cada PR. O workflow
[GitHub Pages](.github/workflows/pages.yml) publica após integração na `main`, com
fonte Pages configurada como **GitHub Actions**. A URL configurada é
`https://joao-kouznetz.github.io/Deeplearnig/`; a publicação depende do merge e do
primeiro deployment bem-sucedido.

## Escolha da base e atribuição

Stroke Prediction é um dos exemplos de classificação do próprio enunciado e foi
selecionado pelo assistente; a escolha específica ainda pode ser confirmada pelo aluno. A página contém uma regra geral de
aprovação e não publica decisões individuais por equipe; o repositório não
inventa comprovante de formulário ou aprovação individual.

Fonte: fedesoriano (2021), [Stroke Prediction Dataset, versão 1](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
Os metadados reservam os direitos dos arquivos aos autores originais e indicam
uso educacional, com origem confidencial. Não atribuímos CC0/CC BY ao arquivo e
não redistribuímos o CSV bruto.
