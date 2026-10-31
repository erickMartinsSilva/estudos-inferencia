import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

mtcars_df = pd.read_csv("./data/mtcars.csv")

automaticos = mtcars_df[mtcars_df["am"] == 0]
manuais = mtcars_df[mtcars_df["am"] == 1]

consumo_automaticos = automaticos["mpg"]
consumo_manuais = manuais["mpg"]

# i.
output_file = Path("./resultados/q2_boxplot.png")
output_file.parent.mkdir(parents=True, exist_ok=True)
fig, (axis1, axis2) = plt.subplots(1, 2, sharey=True)
fig.suptitle("Boxplots do consumo médio por câmbio")

axis1.boxplot(consumo_automaticos, meanline=True)
axis1.set_title("Automático")
axis1.set_ylabel("Consumo (mpg)")
axis1.get_xaxis().set_visible(False)
axis2.boxplot(consumo_manuais, meanline=True)
axis2.set_title("Manual")
axis2.get_xaxis().set_visible(False)

fig.savefig(output_file)
plt.close()

fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, sharey=True)
output_file = Path("./resultados/q2_histogramas.png")
fig.suptitle("Histogramas do consumo médio por câmbio")

axis1.hist(consumo_automaticos, bins="auto")
axis1.set_title("Automático")
axis1.set_xlabel("Consumo (mpg)")
axis1.set_ylabel("Frequência")
axis2.hist(consumo_manuais, bins="auto")
axis2.set_title("Manual")
axis2.set_xlabel("Consumo (mpg)")

fig.savefig(output_file)
plt.close()

# ii. Os procedimentos paramétricos de testes de hipóteses sobre duas amostras que estudamos em nosso curso
# presumem que as amostras a serem usadas são provenientes
# de uma distribuição aproximadamente normal. Para cada uma das amostras fornecidas, verifique essa
# hipótese usando dois dos testes de normalidades que estudamos
# no curso. Para isso, use um nível de significância igual a 5%.

amostras = [
    {
        "nome": "automatico",
        "amostra": consumo_automaticos,
    },
    {
        "nome": "manual",
        "amostra": consumo_manuais,
    }
]

resultados_shapiro = {}
for amostra in amostras:
    # qq plot
    output_file = Path(f"./resultados/q2_qq_plot_{amostra["nome"]}")
    qq = stats.probplot(amostra["amostra"], plot=plt)

    plt.title(f"Q-Q Plot: consumo de carros de câmbio {amostra["nome"]}")
    plt.xlabel("Quantis teóricos")
    plt.ylabel("Consumo (mpg)")
    plt.savefig(output_file)
    plt.close()

    # shapiro-wilk
    alpha = 0.05
    normality_test = stats.shapiro(amostra["amostra"])
    statistic = normality_test.statistic
    pvalue = normality_test.pvalue

    resultados_shapiro[amostra["nome"]] = {
        "statistic": normality_test.statistic,
        "pvalue": normality_test.pvalue,
        "evidencia_contra_normalidade": "Sim" if normality_test.pvalue < alpha else "Nao"
    }

# iii. Presumindo que os dados da coluna mpg seguem a distribuição normal, determine
# um intervalo de confiança no nível 95% da diferença entre as médias dos modelos de
# carros que possuem transmissão automática e os que possuem transmissão manual.
# Assuma variâncias populacionais desconhecidas e não necessariamente iguais.

# IC bilateral de Welch para mu_manual - mu_automatico.
# Amostras independentes: combinar variâncias, sem subtrair observações.
# SciPy calcula erro padrão e graus de liberdade de Welch usando a distribuição t.
confidence_level = 0.95
welch_ci95 = stats.ttest_ind(
    consumo_manuais,
    consumo_automaticos,
    equal_var=False,
    alternative="two-sided"
).confidence_interval(confidence_level=confidence_level)

# iv. Suponha que, ao analisar os dados fornecidos, alguém levantou a alegação de que
# carros automáticos e manuais não apresentam a mesma eficiência relativa a consumo
# de combustível. Em particular, existe a suspeita de que carros com transmissão
# manual apresentam, em média, maior valor de mpg do que carros com transmissão
# automática. Aplique um teste de hipóteses para verificar essa suspeita. Apresente
# e descreva claramente a aplicação dos quatro passos do procedimento. Repare que
# são fornecidas duas amostras independentes. Sendo assim, certifique-se de usar o
# teste de hipóteses adequado para essa situação. Apresente sua análise para o nível
# de significância igual a 5%. Assuma variâncias populacionais desconhecidas e não
# necessariamente iguais.

# duas amostras independentes: t-teste de Welch

# -- DEFINIÇÃO DAS HIPÓTESES --
# hipótese nula: mu_manual <= mu_automatico
# hipótese alternativa: mu_manual > mu_automatico (unicaudal à direita)
alpha = 0.05

# -- APLICAÇÃO DO TESTE --
result = stats.ttest_ind(
    consumo_manuais,
    consumo_automaticos,
    equal_var=False,
    alternative="greater"
)

# -- REALIZAR DECISÃO COM BASE NO P-VALOR --
decision = "Rejeitar h0" if float(result.pvalue) < alpha else "Nao rejeitar h0"

resultados = {
    "n_automatic": len(automaticos),
    "n_manual": len(manuais),
    "normality_tests": resultados_shapiro,
    "welch_ci95": welch_ci95,
    "t_statistic": result.statistic,
    "df": result.df,
    "pvalue": result.pvalue,
    "decision": decision
}

output_file = Path("./resultados/q2_resultados.json")
output_file.write_bytes(json.dumps(resultados, indent=4).encode("utf-8"))
