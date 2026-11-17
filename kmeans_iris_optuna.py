# %% [markdown]
# # Desafio: K-Means + Optuna no conjunto de dados Iris
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score, completeness_score)

# %% 1. Importar o dataset
iris = load_iris()
X = iris.data          # Atributos das flores
y = iris.target        # Classes reais (usadas apenas para avaliação)
df = pd.DataFrame(X, columns=iris.feature_names)

# %% 2. Análise exploratória
print(df.shape, df.isna().sum().sum(), "valores ausentes")
print(df.describe().round(2))
print(df.groupby(y).mean().round(2).set_axis(iris.target_names))

fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for i, c in enumerate(df.columns):
    ax[0, i].hist(df[c], bins=20); ax[0, i].set_title(c, fontsize=9)
    for k, nome in enumerate(iris.target_names):
        ax[1, i].hist(df[c][y == k], bins=15, alpha=.6, label=nome)
ax[1, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("exploracao.png", dpi=110); plt.close()

# %% 3. K-Means base (k=3, sem otimização) para referência
def metricas(rotulos):
    return {"silhouette": silhouette_score(X, rotulos),
            "ARI": adjusted_rand_score(y, rotulos),
            "NMI": normalized_mutual_info_score(y, rotulos),
            "homogeneidade": homogeneity_score(y, rotulos),
            "completude": completeness_score(y, rotulos)}

base = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
print("K-Means base (k=3):", {k: round(v, 3) for k, v in metricas(base.labels_).items()})

# %% 4. Otimização dos hiperparâmetros com Optuna
# O alvo NÃO é usado na otimização (aprendizado não supervisionado): maximiza-se o coeficiente de silhueta.
def objetivo(trial):
    modelo = KMeans(
        n_clusters=trial.suggest_int("n_clusters", 2, 10),
        init=trial.suggest_categorical("init", ["k-means++", "random"]),
        n_init=trial.suggest_int("n_init", 1, 20),
        max_iter=trial.suggest_int("max_iter", 50, 500),
        algorithm=trial.suggest_categorical("algorithm", ["lloyd", "elkan"]),
        random_state=42)
    return silhouette_score(X, modelo.fit_predict(X))

optuna.logging.set_verbosity(optuna.logging.WARNING)
estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
estudo.optimize(objetivo, n_trials=100)
print("Melhores hiperparâmetros:", estudo.best_params)
print("Melhor silhueta:", round(estudo.best_value, 4))

# %% 5. Avaliação com as classes reais
otimo = KMeans(random_state=42, **estudo.best_params).fit(X)
print("K-Means otimizado:", {k: round(v, 3) for k, v in metricas(otimo.labels_).items()})

# Silhueta e ARI por k (mesmos demais parâmetros), para entender a escolha do Optuna
linhas = []
for k in range(2, 11):
    r = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    linhas.append({"k": k, "silhouette": silhouette_score(X, r), "ARI": adjusted_rand_score(y, r)})
por_k = pd.DataFrame(linhas).round(3); print(por_k.to_string(index=False))
print(pd.crosstab(pd.Series(y, name="espécie real").map(dict(enumerate(iris.target_names))),
                  pd.Series(otimo.labels_, name="cluster")))

# Gráficos
pca = PCA(2).fit(X); P = pca.transform(X)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(P[:, 0], P[:, 1], c=y); ax[0].set_title("Espécies reais (PCA)")
ax[1].scatter(P[:, 0], P[:, 1], c=otimo.labels_); ax[1].set_title(f"K-Means otimizado (k={otimo.n_clusters})")
ax[2].plot(por_k["k"], por_k["silhouette"], "o-", label="silhueta"); ax[2].plot(por_k["k"], por_k["ARI"], "s-", label="ARI")
ax[2].set_xlabel("k"); ax[2].legend(); ax[2].set_title("Métricas por número de clusters")
plt.tight_layout(); plt.savefig("resultados.png", dpi=110); plt.close()

plt.figure(figsize=(6, 3.5)); plt.plot([t.number for t in estudo.trials], [t.value for t in estudo.trials], "o", ms=3)
plt.xlabel("trial"); plt.ylabel("silhueta"); plt.title("Histórico da otimização"); plt.tight_layout()
plt.savefig("historico_optuna.png", dpi=110)

# %% [markdown]
# ## 6. Análise dos resultados
# - **Dados:** 150 amostras, 4 atributos em cm, sem valores ausentes. As medidas de pétala separam muito bem as espécies:
#   a setosa é claramente distinta, enquanto versicolor e virginica se sobrepõem.
# - **K-Means base (k=3):** silhueta ≈ 0,55, ARI ≈ 0,73, NMI ≈ 0,76. Recupera bem a setosa e mistura parte de versicolor/virginica (cerca de 16 amostras trocadas).
# - **Otimização com Optuna (maximizando a silhueta):** a escolha tende a k=2 (silhueta ≈ 0,68), pois o critério interno
#   enxerga dois grupos bem separados: setosa de um lado e versicolor+virginica juntas do outro.
#   Contra as espécies reais, esse modelo tem ARI ≈ 0,54: completude alta (≈ 0,88), mas homogeneidade menor (≈ 0,52), porque junta duas espécies.
# - **Conclusão:** uma métrica interna (silhueta) e as classes reais podem discordar. O k "melhor" para a silhueta (2) não é o número
#   de espécies (3), e o ARI é maior em k=3. Como as classes servem só para avaliação, não entram na otimização.
#   Para recuperar as três espécies, pode-se fixar k=3 e otimizar só os demais hiperparâmetros (init, n_init, max_iter); o ganho é pequeno, pois
#   o K-Means já converge para uma solução estável nesse conjunto.
# - **Limitações / melhorias:** K-Means assume grupos esféricos e de tamanho parecido, o que não vale bem para versicolor/virginica;
#   testar padronização dos atributos, Gaussian Mixture ou clustering hierárquico, e usar outra métrica-objetivo (ex.: Calinski-Harabasz ou Davies-Bouldin).
