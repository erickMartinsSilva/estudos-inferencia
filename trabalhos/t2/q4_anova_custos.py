import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data9_df = pd.read_excel("./data/data9.xls")

assuntos = ["math", "comp", "chem"]
labels_assuntos = {
    "math": "Matemática",
    "comp": "Computação",
    "chem": "Química"
}

grupos = {}
grupos_plt = {}
for assunto in assuntos:
    df_assunto = data9_df[data9_df["Cource"].str.startswith(assunto.upper())]
    df_assunto_custos = df_assunto["Cost"]
    grupos[assunto] = {
        "costs": df_assunto_custos,
        "mean": df_assunto_custos.mean()
    }
    grupos_plt[labels_assuntos[assunto]] = df_assunto_custos.sum()


grupos_plt = dict(sorted(grupos_plt.items(), key=lambda item: item[1], reverse=True))
output_path = Path("./resultados/q4_custos.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.bar(list(grupos_plt.keys()), list(grupos_plt.values()))
plt.title("Comparativo de custos por assunto")
plt.xlabel("Assunto")
plt.ylabel("Custo total (R$)")
plt.savefig(output_path)

# buscar evidências de que custos médios de curso não são todos iguais
# h0: mu_1 = mu_2 = mu_3
# hA: pelo menos um par de médias diferente

alpha = 0.01
f_stat, pvalue = stats.f_oneway(
    grupos["comp"]["costs"],
    grupos["math"]["costs"],
    grupos["chem"]["costs"],
)

resultados = {
    "group_sizes": {assunto: len(grupos[assunto]["costs"]) for assunto in assuntos},
    "group_means": {assunto: grupos[assunto]["mean"] for assunto in assuntos},
    "anova_statistic": float(f_stat),
    "anova_pvalue": float(pvalue),
    "alpha": alpha,
    "decision": "Rejeitar h0" if pvalue <= alpha else "Nao rejeitar h0"
}

output_path = Path("./resultados/q4_resultados.json")
output_path.write_bytes(json.dumps(resultados, indent=4).encode("utf-8"))
