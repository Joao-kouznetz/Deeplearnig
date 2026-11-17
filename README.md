**[Acesse o projeto no GitHub Pages →](https://joao-kouznetz.github.io/Deeplearnig/)**

# Deeplearnig

Projetos de análise exploratória e classificação da disciplina de Redes Neurais e Deep
Learning do Insper. Usamos a base **Stroke Prediction** para investigar as
características dos registros e sua relação com a ocorrência de AVC.

No site estão a análise completa, os gráficos e as conclusões. O código e os
resultados usados no relatório ficam neste repositório.

- [Ler o relatório](https://joao-kouznetz.github.io/Deeplearnig/projects/eda/)
- [Como executar o projeto](docs/projects/eda/reproduce.md)
- [Fonte dos dados e atribuição](data/README.md)
- [Projeto 2 — relatório de classificação](docs/projects/classification/index.md)
- [Reproduzir classificação](docs/projects/classification/reproduce.md)
- [Guia de estudo baseado nas aulas](docs/projects/classification/study_guide.md)

## Classificação com PyTorch

A entrega está em `docs/projects/classification/`: rede e loop em `code/`, sete
figuras e tabelas auditáveis. Foram executadas 12 configurações e três seeds finais,
com o mesmo split externo da EDA e seleção exclusiva na validação.
A AP de teste foi **0,1824 ± 0,0177** na MLP e **0,1947** na regressão logística.
O critério de ganho de 10% não foi atingido; a análise registra essa limitação.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python docs/projects/classification/code/experiment.py all --output-dir artifacts/reproduction
python -m pytest -q
python -m mkdocs build --strict
```

O comando escreve os novos resultados em `artifacts/reproduction`, sem substituir
a entrega publicada. Os resultados já presentes podem ser lidos sem treinamento.

## Entrega da EDA

A entrega pedida é o [relatório no GitHub Pages](https://joao-kouznetz.github.io/Deeplearnig/projects/eda/),
com o arquivo [docs/projects/eda/index.md](docs/projects/eda/index.md), código e figuras no repositório.
O formato segue a seção [Submission do enunciado](https://insper.github.io/ann-dl/2026.2/projects/eda/#submission).

Não é exigida entrega em Jupyter Notebook. O [notebook de apoio do professor no Colab](https://colab.research.google.com/drive/1ytcn6p5Jbk-0swDvGmLOV0qnP5oP0fa5?usp=sharing)
é um exemplo de EDA com Palmer Penguins, não o notebook deste projeto.
O enunciado também disponibiliza o [formulário de entrega](https://forms.gle/zREqauuojy8Phvbb7).

Este projeto foi desenvolvido com auxílio de inteligência artificial (OpenAI Codex),
incluindo código, análises, gráficos e redação.
