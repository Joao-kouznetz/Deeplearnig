# Reproduzir o projeto de classificação

Execute na raiz com **Python 3.12**, em um ambiente virtual novo. A execução de
referência usou macOS ARM64, Python 3.12.4, CPU e uma thread. Dependências diretas
estão fixadas em `requirements.txt`; `requirements-lock.txt` registra também
as dependências transitivas instaladas no ambiente limpo. Os metadados estão em
[evaluation.json](tables/evaluation.json).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python docs/projects/classification/code/experiment.py all --output-dir artifacts/reproduction
python -m pytest -q
ruff check docs/projects/eda/code docs/projects/classification/code tests
ruff format --check docs/projects/eda/code docs/projects/classification/code tests
mkdocs build --strict
```

Para fixar também as versões transitivas da execução validada, substitua a linha
de instalação por `python -m pip install -r requirements-lock.txt`.

O primeiro download vem da API oficial do Kaggle, versão 1, com SHA-256 verificado.
Não requer chave para este conjunto público. O CSV bruto fica em `data/raw/`,
fora do Git; EDA e classificação compartilham a aquisição. Se o download falhar,
o programa não troca silenciosamente a fonte. Depois, usa o CSV local validado.

## Fases e regra do teste

`all` executa as três fases abaixo. Alternativamente, use os comandos separados:

```bash
python docs/projects/classification/code/experiment.py tune --output-dir artifacts/reproduction
python docs/projects/classification/code/experiment.py evaluate --output-dir artifacts/reproduction
python docs/projects/classification/code/experiment.py report --output-dir artifacts/reproduction
```

Escolha **`all` ou os três comandos separados**, não ambos na mesma pasta.

1. `tune`: registra o protocolo antes de treinar, confere o split da EDA, ajusta o
   pipeline somente no treino, executa baselines e as 12 candidatas e congela a seleção.
   Não calcula métricas de classificadores no teste.
2. `evaluate`: confere o hash do código/protocolo, retreina a arquitetura escolhida
   nas seeds 42/43/44 e fixa todos os limiares pela validação. Depois avalia todos
   no teste, em um lote. Calcula importância na validação e erros no teste.
3. `report`: lê as tabelas e gera relatório, resumo e sete figuras, sem treinar.

O programa recusa uma busca ou avaliação em uma pasta que já contenha
`tables/evaluation.json`. Para repetir a reprodução, use outra pasta, como
`--output-dir artifacts/reproduction-2`. Isso serve para auditoria; não para
ajustar configurações olhando o teste. Nenhuma seed é escolhida pelo teste.

Sem `--output-dir`, a saída é a pasta da entrega. Ela já contém a avaliação final;
use apenas `report` para regenerar sua apresentação a partir das tabelas.
Checkpoints locais ficam em `checkpoints/`, fora do Git, ou na pasta da reprodução.
Código e CSV oficial permitem refazê-los.

## Auditoria

| Arquivo em `tables/` | O que conferir |
| --- | --- |
| `protocol_before_training.json` | Métrica, critério, busca e seeds antes dos treinos |
| `split.csv` | 3.270 treino, 818 validação, 1.022 teste; índices externos iguais à EDA |
| `baselines_validation.csv` | Baselines e seus limiares de validação |
| `search.csv`, `history_search.csv` | Todas as candidatas e curvas por época |
| `selection.json` | Configuração e baseline selecionados antes do teste |
| `final_before_test.json` | Treinos, épocas e limiares das três seeds congelados |
| `test_metrics.csv`, `test_per_class.csv` | AP, ROC-AUC, confusão e métricas por classe |
| `test_predictions.csv` | Scores e classes para recomputar as métricas |
| `subgroups.csv`, `worst_errors.csv` | Suportes e exemplos da análise de erro |
| `permutation.csv`, `permutation_repeats.csv` | Queda de AP em 20 permutações/atributo na validação |
| `history_final.csv` | Curvas dos retreinos, inclusive depois do melhor checkpoint |
| `evaluation.json`, `results_summary.csv` | Avaliação, versões e resumo da rubrica |

Os testes verificam separação por perfis, reaproveitamento exato do teste da EDA,
ajuste do pipeline só no treino, restauração do checkpoint e recomputação das
métricas pelas predições, além da integridade do relatório/site.
As seeds mudam inicialização e batches; a divisão é sempre 42. Plataforma, BLAS
e versões podem causar pequenas diferenças. Uma versão que altere os índices
externos provoca erro explícito, não uma nova divisão silenciosa.

## Site e entrega

`mkdocs build --strict` gera o site local em `site/`. O menu inclui a classificação
e o guia. A validação de CI reproduz o experimento em pasta separada; Pages
constrói os resultados versionados sem treinar durante a publicação.
Publicar depende de enviar as mudanças para `main`. A execução local não envia
commits, não publica e não preenche formulários.

O [enunciado](https://insper.github.io/ann-dl/2026.2/projects/supervised/) consultado
informa **13 de novembro de 2026, às 23h59** como prazo. Esta implementação foi
executada em **17 de novembro de 2026**; não alegamos entrega no prazo nem
alteramos datas de commits. A situação da entrega deve ser tratada com o professor.

[Voltar ao relatório](index.md).
