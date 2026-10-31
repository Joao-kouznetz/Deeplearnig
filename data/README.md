# Dados, fonte e atribuição

**Stroke Prediction Dataset**, publicado por fedesoriano no Kaggle em 2021,
versão 1. É um dos exemplos de classificação explicitamente linkados no
[enunciado do projeto](https://insper.github.io/ann-dl/2026.2/projects/eda/#0-proposal).

- Fonte: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Metadados: https://www.kaggle.com/api/v1/datasets/view/fedesoriano/stroke-prediction-dataset
- Arquivo: `healthcare-dataset-stroke-data.csv`.
- Dimensões: 5.110 registros × 12 colunas, incluindo `id` e alvo `stroke`.
- Atributos úteis: 3 numéricos e 7 categóricos.
- SHA-256: `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.

`python docs/projects/eda/code/dataset.py` baixa o ZIP oficial com a versão fixada,
lê somente o CSV esperado e verifica o hash antes de gravar `data/raw/`. Cada
leitura posterior repete a checagem. Um arquivo diferente não é aceito
silenciosamente. O endpoint público utilizado não exige credenciais Kaggle.

A fonte atribui os direitos dos arquivos aos autores originais, informa origem
confidencial e solicita uso educacional com crédito ao publicador. Não afirmamos
uma licença CC0 ou CC BY. O CSV bruto não está no Git; os resultados e figuras
foram produzidos para este trabalho educacional e mantêm a atribuição.

O arquivo não traz datas, instituição de coleta, unidade de glicose ou critério
detalhado de rotulagem. `Unknown` em tabagismo é ausência de informação declarada;
`N/A` em IMC é lido como NaN. Os dados não permitem inferir cronologia de doença
ou representatividade populacional.
