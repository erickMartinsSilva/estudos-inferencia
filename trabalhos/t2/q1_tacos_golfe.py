import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

golfe_df = pd.read_csv("data/golfe.csv", sep=';')

# i. 
alpha = 0.05

old_scores = golfe_df["Old Score"]
new_scores = golfe_df["New Score "]

differences = old_scores - new_scores
normality_test = stats.shapiro(differences)
normality_statistic = normality_test.statistic
normality_pvalue = normality_test.pvalue

plot_output_file = Path("./resultados/q1_diferencas.png")
plot_output_file.parent.mkdir(parents=True, exist_ok=True)

fig, (hist_axis, qq_axis) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
fig.suptitle("Diagnóstico das diferenças entre as pontuações")

hist_axis.hist(differences, bins="auto", edgecolor="black")
hist_axis.axvline(0, color="gray", linestyle="--", label="Sem mudança")
hist_axis.set_title("Distribuição")
hist_axis.set_xlabel("Diferença de pontuação")
hist_axis.set_ylabel("Frequência")
hist_axis.legend()

stats.probplot(differences, dist="norm", plot=qq_axis)
qq_axis.set_title("Q-Q plot")
qq_axis.set_xlabel("Quantis teóricos da normal padrão")
qq_axis.set_ylabel("Diferenças ordenadas")

fig.savefig(plot_output_file, dpi=150)
plt.close(fig)

# ii.
# variância desconhecida com amostras pareadas -> t-teste pareado

# hipótese nula: mu_d <= 0 (não há diferença ou pontuação antiga era maior)
# hipótese alternativa: mu_d > 0 (pontuação atual é maior)
alpha = 0.01

result = stats.ttest_rel(
    old_scores,
    new_scores,
    alternative="greater"
)

resultados = {
    "n": len(golfe_df),
    "normality_test": normality_test,
    "normality_statistic": normality_test.statistic,
    "normality_pvalue": normality_test.pvalue,
    "t_statistic": float(result.statistic),
    "df": int(result.df),
    "pvalue": float(result.pvalue),
    "decision": "Rejeitar h0" if float(result.pvalue) < alpha else "Nao rejeitar h0",
}

json_output_file = Path("./resultados/q1_resultados.json")
json_output_file.parent.mkdir(parents=True, exist_ok=True)
json_output_file.write_bytes(json.dumps(resultados, indent=4).encode())


