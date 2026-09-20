import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

gerador = np.random.default_rng(1625)

qtd_simulacoes = 100_000

amostras_x1 = gerador.normal(loc=50, scale=5, size=(qtd_simulacoes, 10))
amostras_x2 = gerador.normal(loc=40, scale=math.sqrt(24), size=(qtd_simulacoes, 8))

media_Md_teorico = 10
variancia_Md_teorico = 5.5
desv_padrao_Md_teorico = math.sqrt(variancia_Md_teorico)

# i.
valores_Md = []
for s in range(qtd_simulacoes):
    Md = amostras_x1[s].mean() - amostras_x2[s].mean()
    valores_Md.append(Md)

valores_Md = np.array(valores_Md)

fig, ax = plt.subplots()
ax.hist(valores_Md, bins="auto")
ax.set_title("Distribuição da diferença entre médias amostrais (X1 - X2)")
ax.set_xlabel("Diferença entre médias")
ax.set_ylabel("Frequência")

output_file = Path("./resultados/q3_md.png")
output_file.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_file)
plt.close()

# ii.
media_empirica_Md = valores_Md.mean()
desv_padrao_empirico_Md = valores_Md.std(ddof=0)

# iii.

z_score = (15 - media_Md_teorico) / desv_padrao_Md_teorico
p_md_maior_que_15_teorico = 1 - norm.cdf(z_score)
 
p_md_maior_que_15_estimativa = np.mean(np.array(valores_Md) > 15)

resultados = {
    "num_simulations": qtd_simulacoes,
    "empirical_mean": media_empirica_Md,
    "empirical_sd": desv_padrao_empirico_Md,
    "theoretical_mean": media_Md_teorico,
    "theoretical_sd": desv_padrao_Md_teorico,
    "probability_ge_15_theoretical": p_md_maior_que_15_teorico,
    "probability_ge_15_empirical": p_md_maior_que_15_estimativa
}

output_file = Path("./resultados/q3_resultados.json")
output_file.parent.mkdir(exist_ok=True, parents=True)
output_file.write_text(json.dumps(resultados, indent=4))

