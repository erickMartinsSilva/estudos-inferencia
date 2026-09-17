import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sturges(n):
    """Calcula a quantidade de bins para histogramas utilizando a regra de Sturges."""
    return math.ceil((np.log2(n)) + 1)

gerador = np.random.default_rng(1625)

P = np.array(range(150, 301))
len_P = len(P)

# pela definição da questão, os elementos do conjunto possuem a mesma probabilidade
prob = (1/len_P) * 100

# i. Gráfico de função de probabilidade de X
probabilidades_df = pd.DataFrame([{ "value": p, "prob": prob } for p in P])

fig, ax = plt.subplots()
ax.bar(probabilidades_df["value"], probabilidades_df["prob"], width=1)
ax.set_title("Gráfico da função de probabilidade de X")
ax.set_xlabel("Quantidade de páginas")
ax.set_ylabel("Probabilidade (%)")
fig.savefig("./q1_populacao.png")
plt.close(fig)

# ii. Medidas centrais e de dispersão
mu = P.mean() 
sigma_squared = P.var()
sigma = math.sqrt(sigma_squared)
print(f"Média populacional: {mu.round(2)}; Variância populacional: {sigma_squared.round(2)}; Desvio padrão amostral: {np.float64(sigma).round(2)}.")

# iii. Construção das amostras
tamanho_amostra_n2 = 2
qtd_amostras_n2 = len_P ** tamanho_amostra_n2
amostras = gerador.choice(P, size=(qtd_amostras_n2, tamanho_amostra_n2), replace=True)

medias_amostras = []
for amostra in amostras:
    media = amostra.mean()
    medias_amostras.append(media)
medias_amostras = np.array(medias_amostras)

fig, ax = plt.subplots()
ax.hist(medias_amostras, bins=sturges(qtd_amostras_n2), linewidth=0.5, edgecolor="white")
ax.set_title("Distribuição das médias amostrais de X\nTamanho de amostra n = 2")
ax.set_xlabel("Média da amostra")
ax.set_ylabel("Frequência")
fig.savefig("./q1_n2.png")
plt.close(fig)

# iv. Medidas amostrais centrais e de dispersão
mu_x = medias_amostras.mean()
sigma_squared_x = medias_amostras.var(ddof=0)
sigma_x = math.sqrt(sigma_squared_x)

print(f"Média amostral: {mu_x.round(2)}; Variância amostral: {sigma_squared_x.round(2)}; Desvio padrão amostral: {np.float64(sigma_x).round(2)}.")

# v. Quantidade de amostras com n = 9
qtd_amostras_possiveis_9 = len_P ** 9 # 40812436757196811351
print(f"Quantidade de amostras de tamanho n = 9 possíveis: {qtd_amostras_possiveis_9}")
# enumeração exaustiva inviável devido à altíssima quantidade de amostras possíveis que podem ser geradas com esse tamanho

# vi. Simulação com 100.000 amostras de tamanho 9
qtd_amostras_n9 = 100_000
amostras_n9 = gerador.choice(P, size=(qtd_amostras_n9, 9), replace=True)
medias_amostras_n9 = []
for amostra in amostras_n9:
    media = amostra.mean()
    medias_amostras_n9.append(media)

fig, ax = plt.subplots()
ax.hist(medias_amostras_n9, bins=sturges(qtd_amostras_n9), linewidth=0.5, edgecolor="white")
ax.set_title("Distribuição das médias amostrais de X\nTamanho de amostra n = 9")
ax.set_xlabel("Média da amostra")
ax.set_ylabel("Frequência")
fig.savefig("./q1_n9.png")
plt.close(fig)

medias_amostras_n9 = np.array(medias_amostras_n9)
mu_n9 = medias_amostras_n9.mean()
mu_n9_teorico = mu
sigma_squared_n9 = medias_amostras_n9.var(ddof=0)
sigma_n9 = math.sqrt(sigma_squared_n9)
sigma_n9_teorico = sigma / 3 # raiz quadrada de 9

resultados_dict = {
    "pop_mean": mu,
    "pop_variance": sigma_squared,
    "pop_sd": sigma,
    "n2_num_samples": qtd_amostras_n2,
    "n2_mean": mu_x,
    "n2_variance": sigma_squared_x,
    "n2_sd": sigma_x,
    "n9_num_possible": qtd_amostras_possiveis_9,
    "n9_num_simulations": qtd_amostras_n9,
    "n9_mean": mu_n9,
    "n9_sd": sigma_n9,
    "n9_theoretical_mean": mu_n9_teorico,
    "n9_theoretical_sd": sigma_n9_teorico
}

with open("./q1_resultados.json", "w") as file:
    file.write(json.dumps(resultados_dict))