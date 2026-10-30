# Dados e atribuição

Usamos o **Online Shoppers Purchasing Intention Dataset**, de C. Sakar e Yomi
Kastro (2018), UCI Machine Learning Repository.

- Página oficial: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- DOI: https://doi.org/10.24432/C5F88Q
- Licença dos dados: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Arquivo: `online_shoppers_intention.csv`, 12.330 linhas, 17 atributos e alvo `Revenue`.
- SHA-256 do CSV original: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

O comando `python docs/projects/eda/code/dataset.py` baixa o ZIP oficial, lê
somente o CSV esperado e confere seu hash **antes** de gravá-lo em `data/raw/`.
O arquivo bruto fica fora do Git. A análise também verifica o hash em cada leitura;
se já estiver baixado, nenhuma conexão com a UCI é necessária.

O CSV não traz identidade de usuário, data completa, moeda de `PageValues` ou
metadados de quando os agregados de Analytics foram calculados. Não inferimos
esses campos. As taxas são frações; mantemos as durações na escala original,
pois o CSV e a página da UCI não explicitam a unidade de tempo. Não há conversão
de moeda ou identificação de pessoas.
