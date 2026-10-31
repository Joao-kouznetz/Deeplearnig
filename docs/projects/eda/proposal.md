# 0. Proposta da base

**Base escolhida: Stroke Prediction**, que aparece explicitamente em
[Examples with the expected profile → Classification, no enunciado](https://insper.github.io/ann-dl/2026.2/projects/eda/#0-proposal).
A escolha segue a orientação do aluno de usar uma das bases indicadas naquela página.

| Campo solicitado | Proposta |
| --- | --- |
| Dataset | Stroke Prediction Dataset, fedesoriano, Kaggle, versão 1 |
| Fonte pública | [Página linkada pelo professor](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Dimensões | 5.110 linhas × 12 colunas |
| Atributos úteis | 10: 3 numéricos e 7 categóricos, além de id e alvo |
| Tarefa | Classificação binária |
| Alvo no arquivo | `stroke`, 0 = ausência de AVC registrado; 1 = AVC registrado |
| Descarte por domínio | `id`, identificador do registro |
| Motivação | Estudar desbalanceamento, ausência de IMC e relações entre atributos mistos e o alvo |
| Primeiro risco | Fonte confidencial e ausência de datas impedem verificar cronologia e representatividade |
| Repositório | [Joao-kouznetz/Deeplearnig](https://github.com/Joao-kouznetz/Deeplearnig) |
| Integrante identificado no Git | João K Bresser |

A base é adequada para uma EDA com variáveis numéricas e categóricas, mais de mil
registros e alvo presente no CSV. As caudas de glicose/IMC e as ausências permitem
justificar decisões concretas de pré-processamento para uma futura rede neural.
A análise é educacional e observacional, sem apresentar associação como causa ou
como estimativa de risco futuro.

A página do trabalho apresenta exemplos e a regra geral de aprovação da proposta;
não publica decisões individuais por equipe. O vínculo com o enunciado está
verificado, mas não inventamos comprovante de envio de formulário ou aprovação
individual. A base será mantida na futura entrega de classificação.

Atribuição: fedesoriano (2021). Os metadados reservam os direitos aos autores
originais e indicam uso educacional, com origem confidencial. O CSV bruto não é
redistribuído; o código faz download oficial da versão 1 com verificação de hash.

**Uso de IA:** proposta elaborada com OpenAI Codex. [Relatório](index.md).
