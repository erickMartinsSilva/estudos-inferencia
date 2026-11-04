import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

compactos = np.array([20 , 25 , 25 , 20 , 22 , 22 , 24 , 26 , 20 , 24 ,
24 , 23 , 20 , 23 , 21 , 21])
medios = np.array([18 , 19 , 22 , 22 , 19 , 16 , 21 , 21 , 21 , 20 ,17 ,
18 , 18 , 17 , 19 , 19 , 18 , 21 , 23 , 19 , 22 , 20])
grandes = np.array([19 , 16 , 16 , 17 , 20 , 20 , 20 , 18 , 18 , 19 , 19 ,
15 , 18 , 17 , 15 , 18 , 17 , 18 , 18 , 17])

alpha = 0.05

# i.
# testes de normalidade
amostras = {
    "compactos": compactos,
    "medios": medios,
    "grandes": grandes
}

output_path = Path("./resultados/q3_grupos.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
axis.boxplot(list(amostras.values()), showmeans=True)
axis.plot([], [], marker="^", color="green", linestyle="none", label="Média")
axis.legend()
axis.set_xticks([1, 2, 3], ["Compactos", "Médios", "Grandes"])
axis.set_title("Comparação de eficiência de combustível por categoria de carro")
axis.set_xlabel("Categoria")
axis.set_ylabel("Eficiência (mpg)")
axis.grid(axis="y", alpha=0.3)
fig.savefig(output_path, dpi=150)
plt.close(fig)

resultados_shapiro = {}
nomes_grupos = {
    "compactos": "Compactos",
    "medios": "Médios",
    "grandes": "Grandes"
}
for chave, amostra in amostras.items():
    # q-q plot
    output_path = Path(f"./resultados/q3_qq_plot_{chave}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    stats.probplot(amostra, plot=axis)
    axis.set_title(f"Q-Q plot normal: {nomes_grupos[chave]}")
    axis.set_xlabel("Quantis teóricos da normal padrão")
    axis.set_ylabel("Eficiência observada ordenada (mpg)")
    axis.legend(axis.get_lines(), ["Observações", "Reta de referência ajustada"])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # shapiro-wilk
    statistic, pvalue = stats.shapiro(amostra)
    resultados_shapiro[chave] = {
        "statistic": statistic,
        "pvalue": pvalue,
        "evidencia_contra_normalidade": "Sim" if pvalue < alpha else "Nao"
    }
# homocedasticidade (variâncias aprox. iguais)
# h0: variâncias iguais
# hA: ao menos um par de variâncias diferentes
levene_stat, levene_pvalue = stats.levene(compactos, medios, grandes)

# ii.
# h0: mu_c = mu_m = mu_g (grupos possuem médias iguais)
# ha: ao menos um par de médias é diferente
anova_stat, anova_pvalue = stats.f_oneway(compactos, medios, grandes)
decision = "Rejeitar h0" if anova_pvalue <= alpha else "Nao rejeitar h0"

resultado = {
    "group_sizes": {
        "compactos": len(compactos),
        "medios": len(medios),
        "grandes": len(grandes)
    },
    "normality_tests": {
        "qq_plot": "consultar arquivos q3_qq_plot na pasta /resultados/",
        "shapiro": resultados_shapiro,
    },
    "levene_statistic": levene_stat,
    "levene_pvalue": levene_pvalue,
    "anova_statistic": anova_stat,
    "anova_pvalue": anova_pvalue,
    "decision": decision
}

output_path = Path("./resultados/q3_resultados.json")
output_path.write_bytes(json.dumps(resultado, indent=4).encode("utf-8"))

