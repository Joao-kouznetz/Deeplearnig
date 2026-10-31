# Conferência com as aulas e exercícios

Esta conferência cobre o material público de ANN-DL 2026.2 consultado em
31/10/2026. Não comprova quais tópicos foram apresentados oralmente em sala.
A entrega é o **Projeto EDA**, não a resolução do exercício de Spaceship Titanic.

| Parte da entrega | Correspondência verificável | Aplicação e limite |
| --- | --- | --- |
| Dicionário, balanceamento, ausentes e estatísticas | [Exercício de dados, 3A](https://insper.github.io/ann-dl/2026.2/exercises/data/#a-get-to-know-the-data) e [tipos](https://insper.github.io/ann-dl/2026.2/classes/data/types/) | O alvo aqui é stroke; os atributos são os da base atual. Não há TotalSpend ou campos de passageiros neste dataset. |
| Separar antes de ajustar | [Exercício 3B](https://insper.github.io/ann-dl/2026.2/exercises/data/#b-split-before-you-transform) e [ordem de operações](https://insper.github.io/ann-dl/2026.2/classes/data/) | Divisão 80/20, seed 42; estatísticas de transformação aprendidas só no treino. |
| Grupos e estratificação | [Aula de splitting](https://insper.github.io/ann-dl/2026.2/classes/data/splitting/) | A implementação combina as duas ideias com StratifiedGroupKFold e fixa o primeiro fold. O grupo representa perfil igual, não identidade comprovada de paciente; proporções são aproximadas. Não é validação cruzada de um modelo. |
| Imputação e indicador | [Qualidade](https://insper.github.io/ann-dl/2026.2/classes/data/quality/) e exercício 3C | Mediana de IMC no treino e flag de ausência. Imputação não recupera o valor verdadeiro nem resolve viés; Unknown continua explícito. |
| One-hot e categoria inédita | [Pré-processamento](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/) e exercício 3C | Vocabulário no treino; handle_unknown=ignore. Os flags são categorias, sem distância ordinal inventada. |
| Imputar → transformar → escalar | [Ordem do pipeline](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/#order-of-operations) | Pipeline + ColumnTransformer; teste recebe somente transform. |
| Log1p | [Transformações](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/transforms/) e exercício 3C–D | Figura 8 mostra antes/depois. Compressão de cauda é constatável; ganho de previsão não foi demonstrado. Comparar com ausência de log apenas em validação interna futura. |
| StandardScaler | [Escala](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/scaling/) e exercício 3C–D | Tabela 17 relata mínimos, máximos, médias e desvios. Padronizar não limita a [-1,1]; o exercício permite padronização ou MinMax. Não se promete ausência de saturação de tanh. |
| PCA e interpretação | [Exercício 2C–D](https://insper.github.io/ann-dl/2026.2/exercises/data/) e [redução](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/reduction/) | Variância, pesos e projeção no treino. Sobreposição em 2D não prova inseparabilidade no espaço original; mesma direção dos pesos não prova correlação causal. |
| t-SNE, UMAP e trustworthiness | [Redução](https://insper.github.io/ann-dl/2026.2/classes/data/preprocessing/reduction/) e [Projeto 4B](https://insper.github.io/ann-dl/2026.2/projects/eda/#4-preprocessing) | Dois parâmetros por método, mesma amostra, alvo somente para cor. A aula usa trustworthiness; os mapas não são entradas de um classificador nesta entrega. |
| Correlações, frequências e boxplots | [Projeto 2–3](https://insper.github.io/ann-dl/2026.2/projects/eda/) | Três numéricas e sete categóricas; Spearman com pares observados, denominadores e conclusão por figura. |
| Nenhum modelo treinado | [Projeto EDA](https://insper.github.io/ann-dl/2026.2/projects/eda/) | Ajustam-se apenas preparação e projeções. Referências a métricas são plano para a próxima entrega, não resultados de modelo. |

## Complementos e escolhas de implementação

Os intervalos de Wilson são um diagnóstico adicional de incerteza, sem alegar
que foram ensinados no exercício consultado ou que façam parte da rubrica.
Hash dos perfis, SHA-256 do arquivo, testes automatizados e CI são mecanismos de
integridade e reprodução. A seleção exata dessas ferramentas não é atribuída ao
professor. A decisão de usar uma categoria explícita para ausência, em vez de
moda, preserva o significado de Unknown nesta fonte e está justificada no relatório.

One-hot completo gera colunas linearmente dependentes por bloco; isso é esperado.
PCA por SVD lida com essa dependência, e suas últimas componentes podem ter
variância praticamente zero. Os indicadores ficam em 0/1 para não aumentar o peso
de categorias raras. Essa ponderação é uma escolha explícita, não uma regra universal.

## Correções desta revisão

- Acrescentar a comparação antes/depois e as faixas finais da padronização.
- Corrigir a leitura dos sinais de pesos da PCA: contribuição ao eixo não é prova
  de correlação entre as medidas originais.
- Explicitar que simetria do histograma não demonstra ganho preditivo.
- Corrigir a atribuição da escolha da base: Stroke foi selecionada pelo assistente
  dentre os exemplos do enunciado, sem confirmação específica do aluno até aqui.
- Verificar os números contra o treino, além de conferir apenas shape e finitude.

[Relatório](index.md). **Uso de IA:** revisão e redação com OpenAI Codex.

As projeções desta entrega são diagnósticos de EDA, ajustados só no treino e
mantidos fora do pipeline final de 24 colunas. Se PCA for usada como entrada de
um modelo posterior, ela deverá entrar no mesmo Pipeline do modelo e ser
reajustada em cada fold. Os mapas atuais não podem ser reaproveitados como
pré-processamento global antes de validação cruzada. Não treinamos uma rede
para reproduzir os exemplos demonstrativos das aulas: isso contrariaria o
escopo desta entrega de EDA.
