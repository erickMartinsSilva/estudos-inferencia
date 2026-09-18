import json
import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gerador = np.random.default_rng(1625)

P = np.array(range(150, 301))
len_P = len(P)

# elementos possuem a mesma probabilidade
prob = (1/len_P) * 100

# i. Gráfico de função de probabilidade de X
probabilidades_df = pd.DataFrame([{ "value": p, "prob": prob } for p in P])

fig, ax = plt.subplots()
ax.bar(probabilidades_df["value"], probabilidades_df["prob"], width=1)
ax.set_title("Gráfico da função de probabilidade de X")
ax.set_xlabel("Quantidade de páginas")
ax.set_ylabel("Probabilidade (%)")

output_file = Path("./resultados/q1_populacao.png")
output_file.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(output_file)
plt.close(fig)

# ii. Medidas centrais e de dispersão
mu = P.mean() 
sigma_squared = P.var()
sigma = math.sqrt(sigma_squared)

# iii. Construção de amostras de tamanho 2
tamanho_amostra_n2 = 2
qtd_amostras_n2 = len_P ** tamanho_amostra_n2
amostras = np.array(list(product(P, repeat=tamanho_amostra_n2)))

medias_amostras = []
for amostra in amostras:
    media = amostra.mean()
    medias_amostras.append(media)
medias_amostras = np.array(medias_amostras)

fig, ax = plt.subplots()
ax.hist(medias_amostras, bins="fd", linewidth=0.5, edgecolor="white")
ax.set_title("Distribuição das médias amostrais de X\nTamanho de amostra n = 2")
ax.set_xlabel("Média da amostra")
ax.set_ylabel("Frequência")

output_file = Path("./resultados/q1_n2.png")
fig.savefig(output_file)
plt.close(fig)

# iv. Medidas amostrais centrais e de dispersão
mu_x = medias_amostras.mean()
sigma_squared_x = medias_amostras.var(ddof=0)
sigma_x = math.sqrt(sigma_squared_x)

# v. Quantidade de amostras com n = 9
tamanho_amostra_n9 = 9
qtd_amostras_possiveis_9 = len_P ** tamanho_amostra_n9

# vi. Simulação com 100.000 amostras de tamanho 9
qtd_amostras_n9 = 100_000
amostras_n9 = gerador.choice(P, size=(qtd_amostras_n9, tamanho_amostra_n9), replace=True)
medias_amostras_n9 = []
for amostra in amostras_n9:
    media = amostra.mean()
    medias_amostras_n9.append(media)

fig, ax = plt.subplots()
ax.hist(medias_amostras_n9, bins="fd", linewidth=0.5, edgecolor="white")
ax.set_title("Distribuição das médias amostrais de X\nTamanho de amostra n = 9")
ax.set_xlabel("Média da amostra")
ax.set_ylabel("Frequência")

output_file = Path("./resultados/q1_n9.png")
fig.savefig(output_file)
plt.close(fig)

medias_amostras_n9 = np.array(medias_amostras_n9)
mu_n9 = medias_amostras_n9.mean()
mu_n9_teorico = mu
sigma_squared_n9 = medias_amostras_n9.var(ddof=0)
sigma_n9 = math.sqrt(sigma_squared_n9)
sigma_n9_teorico = sigma / 3

resultados = {
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

output_file = Path("./resultados/q1_resultados.json")
output_file.write_text(json.dumps(resultados, indent=4))
