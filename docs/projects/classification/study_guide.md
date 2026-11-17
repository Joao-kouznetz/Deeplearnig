# Entender o projeto com as aulas

Este guia conecta o experimento ao curso. Leia junto com o [relatório](index.md)
e execute o [código](code/experiment.py): explicar decisões exige entender de onde
vêm os números, inclusive quando a rede perde.

## Mapa entre aulas e implementação

| Aula | Conceito aplicado | Onde aparece |
| --- | --- | --- |
| [1. Conceitos](https://insper.github.io/ann-dl/2026.2/classes/concepts/) | Aprendizado supervisionado | `stroke` é alvo binário conhecido |
| [2. Dados](https://insper.github.io/ann-dl/2026.2/classes/data/) | Tipos, imputação, transformação e divisão | `common.py` importa o pipeline da EDA e separa validação |
| [3. Redes neurais](https://insper.github.io/ann-dl/2026.2/classes/ann/) | Somas ponderadas e bias | `nn.Linear` em `network.py` |
| [4. Perceptron](https://insper.github.io/ann-dl/2026.2/classes/perceptron/) | Limites de uma fronteira linear | A logística é a referência linear comparada à MLP |
| [5. MLP](https://insper.github.io/ann-dl/2026.2/classes/mlp/) | Camadas ocultas e backpropagation | Rede 24 → 16 → 1; `loss.backward()` |
| [6. Otimização](https://insper.github.io/ann-dl/2026.2/classes/optimization/) | Gradientes e learning rate | Adam e três learning rates na busca |
| [7. Regularização](https://insper.github.io/ann-dl/2026.2/classes/regularization/) | Sobreajuste e controle de capacidade | L2, dropout, early stopping e melhor checkpoint |
| [8. Métricas](https://insper.github.io/ann-dl/2026.2/classes/metrics/classification/) | Ranking, limiar e classes raras | AP, ROC, F2, confusão e subgrupos |

Não precisamos de CNN, atenção ou geração para aplicar os conceitos centrais a
estes dados tabulares. Usar uma arquitetura maior exigiria uma hipótese e validação.

## Caminho de um exemplo

1. Entram três medidas e sete categorias. `id` é descartado; `stroke` é a resposta.
2. IMC ausente recebe a mediana do treino e um indicador de ausência. Glicose/IMC
   passam por `log1p`; medidas são padronizadas e categorias recebem one-hot.
3. As 24 entradas chegam à camada oculta: `h = ReLU(Wx + b)`.
4. A saída é o logit `z = v·h + c`; na inferência, o score é `sigmoid(z)`.
5. Comparamos ao limiar de validação: 0,622840 na seed 42. O score não foi calibrado
   para representar probabilidade populacional.

## Como contar parâmetros

Na rede 24 → 16 → 1, a primeira camada tem 24 × 16 pesos + 16 biases = 400;
a saída tem 16 × 1 pesos + 1 bias = 17. Total: **417 parâmetros** para
**3.270 registros de treino**, dos quais só **159 são positivos**.
Essa razão não garante generalização: as entradas são correlacionadas e os positivos, raros.

## O que acontece em um batch

`zero_grad()` limpa os gradientes acumulados. O `forward` calcula logits;
`BCEWithLogitsLoss` compara logits e alvos, aumentando o peso dos positivos.
`backward()` usa a regra da cadeia para calcular gradientes, e `step()` usa Adam
para atualizar os parâmetros. O próximo batch encontra outros pesos.

Na validação usamos `eval()` para desativar dropout e `no_grad()` para não
armazenar o grafo de gradientes. A época seguinte volta a `train()`.

## Perguntas para explicar na apresentação

**Por que não acurácia?** No teste, prever 0 sempre acerta 972/1.022 = 95,11%,
mas perde todos os 50 positivos. AP mede a priorização dos positivos no ranking.

**Loss e métrica são iguais?** BCE ponderada é diferenciável e orienta o gradiente;
AP compara configurações. Reduzir BCE não garante melhorar AP em toda época.

**Peso de classe e L2 são iguais?** O primeiro aumenta a contribuição dos erros
positivos; L2 penaliza os parâmetros. O peso de classe aqui é 3.111/159 ≈ 19,5660.

**Por que o limiar muda?** A preferência por recall, pesos de classe e calibração
afetam a decisão. Maximizamos F2 na validação, nunca no teste. F2 tem `4FN + FP`
no denominador: a preferência é educacional, sem custo clínico conhecido.

**Onde aparece sobreajuste?** Veja a Figura 1 e as épocas: quando a validação deixa
de melhorar enquanto o treino melhora, guardar o último estado pode piorar a
generalização. O código restaura o melhor checkpoint.

**Dropout e L2 ajudaram?** As quatro linhas da ablação mantêm largura, learning rate,
split e seed. L2 melhorou AP nessa comparação; dropout sozinho não. Entre todas as
candidatas, a rede escolhida tem largura 16 e sem L2/dropout explícitos. Isso não
torna as técnicas inúteis em outros casos; early stopping continua sendo usado.

**Por que três seeds?** Mudam a inicialização e ordem dos batches. As APs foram
0,164654; 0,182388; 0,200113. Média ± desvio amostral: **0,182385 ± 0,017730**.
Isso mede variação de treino no mesmo split; não é intervalo de confiança ou
evidência sobre outra população. Não escolhemos a seed vencedora no teste.

**A rede ganhou?** Na validação sim; no teste, a média ficou abaixo da logística
(0,194683). A meta de +10% exigiria 0,214151. Não foi atingida. Um resultado
negativo responde à pergunta do projeto; a rede não precisa vencer para a comparação ser válida.

**O que é importância por permutação?** Trocamos a ordem de uma variável na validação
e observamos a queda de AP. A maior queda veio da idade. Isso mostra dependência
do modelo naquele conjunto, sem demonstrar causalidade ou efeito clínico.

**Como pode haver erro baixo em jovens e recall zero?** Abaixo de 45 anos há só
cinco positivos em 521 registros. O modelo perde os cinco, mas erra só seis linhas
ao todo. O denominador da taxa de erro esconde a falha que o recall mostra.

## Conferir os números na prática

Filtre `tables/test_predictions.csv` por MLP/seed 42 e conte TP, FP, FN e TN:
**32, 186, 18 e 786**. Assim:

- Precisão: 32/(32+186) = **14,68%**.
- Recall: 32/(32+18) = **64%**.
- F1 positivo: 64/(64+186+18) = **0,2388**.

Compare com a Figura 4. Para AP, ordene por score e observe a precisão nos pontos
em que novos positivos são recuperados. `average_precision_score` soma essas
precisões ponderadas pelos incrementos de recall, sem usar o limiar de classificação.

[Voltar ao relatório](index.md).
