import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gerador = np.random.default_rng(1625)


def estimador(M, k):
    """
    Estimador utilizado na questão 2.

    Args:
        M: Máximo amostral
        k: Tamanho da amostra
    Returns:
        Estimativa pontual para a amostra.
    """
    return M + (M / k) - 1

# i.
amostra = [17, 32, 45, 59, 88]
k = 5
M = 88

estimativa_pontual_N = estimador(M, k)

# ii.
populacao = np.arange(1, 101)
qtd_simulacoes = 10_000

estimativas = []
for _ in range(qtd_simulacoes):
    amostra = gerador.choice(populacao, size=k, replace=False)
    M = max(amostra)
    estimativas.append(estimador(M, k))

# iii.
fig, ax = plt.subplots()
ax.hist(estimativas, bins="fd", linewidth=0.5, edgecolor="white")
ax.set_title(f"Distribuição das estimativas pontuais da população\nTamanho de amostra k = {k}")
ax.set_xlabel("Estimativas pontuais")
ax.set_ylabel("Frequência")

output_file = Path("./resultados/q2_k5.png")
output_file.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_file)
plt.close()

estimativas = np.array(estimativas)
media_estimativas = estimativas.mean()

N = 100
vies = media_estimativas - N

proporcao_subestimacao = np.mean(estimativas < N)
proporcao_igualdade = np.mean(estimativas == N)
proporcao_superestimacao = np.mean(estimativas > N)

# iv.
variancia_empirica_estimativas = estimativas.var(ddof=0)
desv_padrao_empirico_estimativas = estimativas.std(ddof=0)


# v.
valores_k = [2, 5, 10, 20]
by_k = {}
for k_atual in valores_k:
    estimativas = []

    for _ in range(qtd_simulacoes):
        amostra = gerador.choice(populacao, size=k_atual, replace=False)
        estimativas.append(estimador(max(amostra), k_atual))

    estimativas_np = np.array(estimativas)
    media = estimativas_np.mean()
    vies = media - N
    variancia = estimativas_np.var(ddof=0)
    desv_padrao = estimativas_np.std(ddof=0)
    diff_valor_N_quadrado_acc = 0
    
    for e in estimativas_np:
        diff_valor_N_quadrado_acc += (e - N) ** 2
    rmse = math.sqrt((1/qtd_simulacoes) * diff_valor_N_quadrado_acc)

    proporcao_subestimacao = np.mean(estimativas_np < N)
    proporcao_igualdade = np.mean(estimativas_np == N)
    proporcao_superestimacao = np.mean(estimativas_np > N)

    # vi.
    mc_inferior, mc_superior = np.quantile(
        estimativas_np,
        [0.025, 0.975],
        axis=0,
        method="linear"
    )

    by_k[k_atual] = {
        "mean": media,
        "bias": vies,
        "variance": variancia,
        "sd": desv_padrao,
        "rmse": rmse,
        "underestimate_rate": proporcao_subestimacao,
        "equal_rate": proporcao_igualdade,
        "overestimate_rate": proporcao_superestimacao,
        "central_range95": (mc_inferior, mc_superior),
        "contains_true_N": bool(mc_inferior <= N and mc_superior >= N)
    }


by_k_df = pd.DataFrame.from_dict(by_k, orient="index").sort_index()
by_k_df.index.name = "k"

# Gráfico de comparação das estatísticas
estatisticas = [
    ("mean", "Média da estimativa"),
    ("bias", "Viés"),
    ("variance", "Variância"),
    ("sd", "Desvio padrão"),
    ("rmse", "RMSE"),
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(11, 7),
    sharex=True,
    constrained_layout=True
)
axes = axes.ravel()

for ax, (coluna, titulo) in zip(axes, estatisticas):
    ax.plot(
        by_k_df.index,
        by_k_df[coluna],
        marker="o",
        linewidth=2
    )
    ax.set_title(titulo)
    ax.set_ylabel("Valor")
    ax.set_xticks(by_k_df.index)
    ax.grid(axis="y", alpha=0.3)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if coluna == "mean":
        ax.axhline(N, color="red", linewidth=1, linestyle="--")

    if coluna == "bias":
        ax.axhline(0, color="red", linewidth=1, linestyle="--")

for ax in axes:
    ax.set_xlabel("Tamanho da amostra (k)")

axes[-1].set_visible(False)

fig.suptitle("Comparação das métricas das estimativas em função de k")
output_file = Path("./resultados/q2_comparacao_k.png")
plt.savefig(output_file)
plt.close(fig)

resultados = {
    "sample_estimate": estimativa_pontual_N,
    "num_simulations": qtd_simulacoes,
    "by_k": by_k
}

output_file = Path("./resultados/q2_resultados.json")
output_file.write_text(json.dumps(resultados, indent=4))
