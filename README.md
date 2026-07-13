# Sistema Inteligente de Classificação de Diabetes

Este projeto implementa um fluxo completo de machine learning para classificar diabetes a partir de sinais clínicos. A solução foi organizada em três etapas principais:

1. Pré-processamento e análise exploratória.
2. Treinamento e comparação de modelos clássicos.
3. Interface web para fazer previsões em tempo real.

## Arquitetura

- [app.py](app.py): interface Streamlit para uso do usuário.
- [src/preprocess_diabetes.py](src/preprocess_diabetes.py): limpeza, imputação, padronização e divisão do dataset.
- [src/train_models.py](src/train_models.py): treinamento dos modelos e geração das métricas.
- [src/artifact_utils.py](src/artifact_utils.py): caminhos compartilhados para os artefatos gerados.

## Como executar

```bash
python src/preprocess_diabetes.py
python src/train_models.py
streamlit run app.py
```

## Modelos avaliados

Os modelos treinados incluem:

- Árvores de decisão com diferentes profundidades.
- Naive Bayes Gaussiano.

Os resultados são salvos em [models/trained/model_metrics.json](models/trained/model_metrics.json) e em [models/trained/model_report.md](models/trained/model_report.md).
