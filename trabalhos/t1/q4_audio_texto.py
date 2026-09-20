import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data = pd.read_csv("./data/SchroederEpley2015data.txt")
data = data[["CONDITION", "Intellect_Rating"]].dropna()

Sa = data[data["CONDITION"] == 1]
St = data[data["CONDITION"] == 0]

# i.
datasets = [St, Sa]
condition_info = {}
for idx, set in enumerate(datasets):
    intellect_ratings = set["Intellect_Rating"]

    tamanho = len(intellect_ratings)
    media = intellect_ratings.mean()
    desv_padrao = intellect_ratings.std(ddof=1)

    # iii.
    stat_set, p_set = stats.shapiro(intellect_ratings)

    # iv.
    n = len(set)
    erro_padrao_t = desv_padrao / math.sqrt(n)
    graus_liberdade = n - 1
    t_critico = stats.t.ppf(0.995, df=graus_liberdade)
    margem_erro_t = t_critico * erro_padrao_t
    IC = (media - margem_erro_t, media + margem_erro_t)

    condition_info[idx] = {
        "tamanho": tamanho,
        "media": media,
        "desv_padrao": desv_padrao,
        "shapiro_stat": stat_set,
        "shapiro_p_value": p_set,
        "intervalo_99": IC
    }

# ii.
fig, ax = plt.subplots()
ax.boxplot([
    Sa["Intellect_Rating"],
    St["Intellect_Rating"]
])
ax.set_title("Avaliação do intelecto por condição")
ax.set_xticks([1, 2], ["Áudio", "Transcrição"])
ax.set_ylabel("Taxa de Intelecto")

output_file = Path("./resultados/q4_boxplot.png")
output_file.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_file)
plt.close()

# v.
Sa_ratings = Sa["Intellect_Rating"]
St_ratings = St["Intellect_Rating"]

n_Sa = len(Sa_ratings)
n_St = len(St_ratings)
media_Sa = Sa_ratings.mean()
media_St = St_ratings.mean()
variancia_Sa = Sa_ratings.var(ddof=1)
variancia_St = St_ratings.var(ddof=1)

diferenca_medias = media_Sa - media_St
erro_padrao_welch = math.sqrt(
    variancia_Sa / n_Sa + variancia_St / n_St
)

graus_liberdade_welch = (
    (variancia_Sa / n_Sa + variancia_St / n_St) ** 2
    / (
        (variancia_Sa / n_Sa) ** 2 / (n_Sa - 1)
        + (variancia_St / n_St) ** 2 / (n_St - 1)
    )
)

t_critico_welch = stats.t.ppf(0.975, df=graus_liberdade_welch)
margem_erro_welch = t_critico_welch * erro_padrao_welch
IC_welch_95 = (
    diferenca_medias - margem_erro_welch,
    diferenca_medias + margem_erro_welch
)

resultados = {
    "n_audio": condition_info[1]["tamanho"],
    "mean_audio": condition_info[1]["media"],
    "sd_audio": condition_info[1]["desv_padrao"],
    "n_transcription": condition_info[0]["tamanho"],
    "mean_transcription": condition_info[0]["media"],
    "sd_transcription": condition_info[0]["desv_padrao"],
    "shapiro_audio": {
        "statistic": condition_info[1]["shapiro_stat"],
        "p_value": condition_info[1]["shapiro_p_value"],
        "reject_normality_at_0_05": bool(
            condition_info[1]["shapiro_p_value"] < 0.05
        )
    },
    "shapiro_transcription": {
        "statistic": condition_info[0]["shapiro_stat"],
        "p_value": condition_info[0]["shapiro_p_value"],
        "reject_normality_at_0_05": bool(
            condition_info[0]["shapiro_p_value"] < 0.05
        )
    },
    "ci99_audio": [
        condition_info[1]["intervalo_99"][0],
        condition_info[1]["intervalo_99"][1]
    ],
    "ci99_transcription": [
        condition_info[0]["intervalo_99"][0],
        condition_info[0]["intervalo_99"][1]
    ],
    "ci95_difference_welch": [
        IC_welch_95[0],
        IC_welch_95[1]
    ]
}

output_file = Path("./resultados/q4_resultados.json")
output_file.parent.mkdir(exist_ok=True, parents=True)
output_file.write_text(
    json.dumps(resultados, indent=4),
)
