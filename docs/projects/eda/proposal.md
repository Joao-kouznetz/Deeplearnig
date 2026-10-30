# 0. Proposta da base

**Status: preparada; envio ao formulário e aprovação docente não comprovados.**
A base não foi apresentada como já aprovada. O enunciado permite propor até três
opções; escolhemos uma e documentamos o risco principal, sem criar alternativas
que não serão analisadas.

| Campo solicitado | Proposta |
| --- | --- |
| Dataset | Online Shoppers Purchasing Intention Dataset |
| Fonte pública | [UCI Machine Learning Repository, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| DOI | [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q) |
| Dimensões originais | 12.330 linhas × 18 colunas: 17 atributos e 1 alvo |
| Tarefa | Classificação binária |
| Alvo presente no arquivo | `Revenue`: sessão com compra (True) ou sem compra (False) |
| Atributos numéricos | 10; 9 retidos após excluir PageValues |
| Atributos categóricos | 7; códigos numéricos nominais tratados como categorias |
| Licença | CC BY 4.0, com atribuição a C. Sakar e Yomi Kastro (2018) |
| Repositório da equipe | [Joao-kouznetz/Deeplearnig](https://github.com/Joao-kouznetz/Deeplearnig) |
| Integrante identificado na configuração Git | João K Bresser |

As sessões permitem investigar como volume e duração de navegação, mês e tipo de
visitante se associam à ocorrência de compra. O desbalanceamento, as caudas longas
e os atributos mistos tornam a base adequada para estudar preparação para uma
rede neural, com porte suficiente para comparações de redução de dimensionalidade.

**Primeiro risco:** `PageValues` depende da atribuição de valor a páginas visitadas
antes de transações; o arquivo não comprova sua disponibilidade anterior ao
desfecho. Excluímos essa coluna do pipeline por precaução. `BounceRates` e
`ExitRates` também dependem de agregados históricos cuja janela não é informada;
o escopo inicial é retrospectivo, sem alegar previsão durante a sessão.

O aluno precisa confirmar com o professor se esta base já foi usada na turma e
obter sua aprovação. O tamanho e o caráter público são verificáveis, mas não
permitem inferir aprovação acadêmica. A mesma base deverá continuar na próxima
entrega de classificação.

[Enunciado e formulário indicados pela disciplina](https://insper.github.io/ann-dl/2026.2/projects/eda/).

**Uso de IA:** proposta elaborada com OpenAI Codex. [Voltar ao relatório](index.md).
