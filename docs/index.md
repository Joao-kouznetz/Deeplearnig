# Deeplearnig

Portfólio de João K Bresser para Redes Neurais Artificiais e Deep Learning,
Insper, 2026.2.

## Projeto 1 — EDA

O [relatório](projects/eda/index.md) analisa o **Stroke Prediction Dataset**,
uma das bases de classificação indicadas no enunciado. O alvo é `stroke` (0/1).
A entrega cobre qualidade, ausentes, distribuições, associações, PCA, t-SNE, UMAP
e pipeline ajustado somente no treino. Não há treinamento de classificador.

- [Proposta e vínculo com o enunciado](projects/eda/proposal.md)
- [Reproduzir a análise](projects/eda/reproduce.md)
- [Repositório](https://github.com/Joao-kouznetz/Deeplearnig)

## Projeto 2 — Classificação supervisionada

O [relatório de classificação](projects/classification/index.md) continua a EDA
com uma MLP em PyTorch, baselines, 12 configurações na validação, regularização,
early stopping e avaliação final em três seeds. A AP da rede no teste foi
**0,1824 ± 0,0177**, contra **0,1947** da regressão logística: a rede não atingiu
a meta pré-fixada. O relatório discute o resultado e os erros por subgrupo.

- [Reproduzir o experimento](projects/classification/reproduce.md)
- [Entender as decisões com as aulas](projects/classification/study_guide.md)

**Uso de IA:** projeto desenvolvido com inteligência artificial (OpenAI Codex),
incluindo código, análise, gráficos, testes e redação. Os números foram calculados
pela execução do código. O uso está declarado também em `ai_use` no relatório.
