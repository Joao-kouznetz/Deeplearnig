# Deeplearnig — Projeto EDA

Análise exploratória do **Online Shoppers Purchasing Intention Dataset** (UCI),
para o [Projeto 1 de ANN-DL 2026.2 do Insper](https://insper.github.io/ann-dl/2026.2/projects/eda/).
Tarefa: classificação binária de `Revenue`, indicando se uma sessão terminou em compra.
Esta entrega prepara os dados para modelagem posterior; não treina um classificador.

**Este projeto foi feito usando inteligência artificial (OpenAI Codex)**, incluindo
seleção dos dados, implementação, gráficos, testes, análise e redação. Os números
foram calculados pela execução do código. O uso de IA também está declarado no
front matter do relatório. A revisão humana e a capacidade de explicar as decisões
continuam necessárias; não atribuímos a análise a uma pessoa que não a revisou.

- [Relatório completo, na estrutura da rubrica](docs/projects/eda/index.md)
- [Proposta da base e status de aprovação](docs/projects/eda/proposal.md)
- [Instruções detalhadas de reprodução](docs/projects/eda/reproduce.md)
- [Código importável do pipeline](docs/projects/eda/code/preprocessing.py)
- [Tabelas e resultados auditáveis](docs/projects/eda/tables/)
- [Fonte, licença e integridade dos dados](data/README.md)

## Executar a partir de um clone limpo

Requer **Python 3.12** e acesso à internet para instalar as dependências e baixar
o CSV oficial na primeira execução. O ambiente foi validado com Python 3.12.16;
todas as dependências estão fixadas em `requirements.txt`.

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

Se `venv` não estiver disponível no sistema, uma alternativa com
[uv](https://docs.astral.sh/uv/) já instalado é:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python docs/projects/eda/code/analyze.py
.venv/bin/python -m pytest -q
.venv/bin/mkdocs build --strict
```

O comando principal baixa e verifica os dados, faz a divisão com seed 42, calcula
as estatísticas no treino, ajusta o pipeline, gera nove figuras numeradas e
reescreve o relatório com os resultados efetivamente calculados. As execuções de
t-SNE/UMAP e a primeira compilação do Numba podem levar alguns minutos. Não é
necessária conta Kaggle, chave de API ou edição manual de caminho.

Os arquivos de entrega (relatório, figuras e tabelas) estão versionados. Dados
brutos, ambiente, cache, site compilado e o pipeline binário regenerável em
`artifacts/preprocessor.joblib` ficam fora do Git. Após o download inicial, a
análise pode ser reexecutada sem conexão com a UCI.

## Decisões que evitam interpretações enganosas

- `PageValues` é excluído preventivamente: não há prova de que seu valor foi
  calculado antes da transação. O alvo também fica fora da matriz de entrada.
- As 125 duplicatas exatas excedentes são removidas; perfis de entrada iguais
  ficam no mesmo lado da divisão entre treino e teste.
- Medianas, escalas, categorias, correlações e projeções são calculadas no treino.
  O teste recebe apenas a transformação já ajustada.
- Caudas longas recebem `log1p`; nenhum registro é removido por ser extremo.
- PCA, duas configurações de t-SNE e duas de UMAP são comparadas na mesma amostra
  de treino. Ilhas visuais não são tratadas como prova de classes separáveis.

## Site e integração contínua

A documentação usa MkDocs Material. `mkdocs build --strict` gera `site/index.html`.
O workflow [Validar EDA](.github/workflows/validate.yml) reproduz a análise,
executa os testes e oferece o site compilado como artefato `site-eda` em cada PR.
O workflow [GitHub Pages](.github/workflows/pages.yml) publica automaticamente
depois que as alterações estiverem na `main`, usando Pages com fonte
**GitHub Actions**. A URL configurada é
`https://joao-kouznetz.github.io/Deeplearnig/`; isso não afirma que o site já esteja
publicado antes do merge e do primeiro deployment bem-sucedido.

## Pendência acadêmica

A base está proposta, **sem aprovação docente comprovada**. O enunciado exige
essa aprovação e o cadastro do repositório; os formulários dependem do aluno e do
professor. O repositório contém a proposta pronta para revisão, sem inventar
envio, aprovação, integrantes ou data de entrega anterior ao commit real.

## Referência e licença dos dados

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*.
UCI Machine Learning Repository. [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
Dados sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
