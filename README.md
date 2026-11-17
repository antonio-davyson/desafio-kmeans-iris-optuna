# Agrupamento do dataset Iris com K-Means e Optuna

Desafio Rocketseat: agrupar as amostras do conjunto Iris com K-Means, otimizando os hiperparâmetros com Optuna
e avaliando os clusters com as espécies reais.

## Conteúdo
- `kmeans_iris_optuna.py` — importação, exploração, K-Means, otimização, avaliação e análise (no final do arquivo)
- `exploracao.png`, `resultados.png`, `historico_optuna.png` — gráficos gerados

## Como executar
```bash
pip install -r requirements.txt
python kmeans_iris_optuna.py
```

## Abordagem
- O Optuna maximiza o coeficiente de silhueta (métrica interna); as classes reais só são usadas na avaliação (ARI, NMI, homogeneidade, completude).
- Hiperparâmetros otimizados: `n_clusters` (2–10), `init`, `n_init`, `max_iter`, `algorithm`.

## Principais achados
| Modelo | Silhueta | ARI |
|---|---|---|
| K-Means base (k=3) | 0,553 | 0,730 |
| k=2 (preferido pela silhueta) | 0,681 | 0,540 |

A silhueta favorece k=2 (setosa vs. o resto), enquanto as espécies reais correspondem a k=3. Detalhes na análise no final do código.
