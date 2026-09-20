import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

data = pd.read_csv("./data/survey.csv")["Height"].dropna()

# i.
tamanho = len(data)
media = data.mean()
desv_padrao = data.std(ddof=1)

# ii.
erro_padrao_t = desv_padrao / math.sqrt(tamanho)
graus_liberdade = tamanho - 1
t_critico = stats.t.ppf(0.975, df=graus_liberdade)
margem_erro_t = t_critico * erro_padrao_t
IC = (media - margem_erro_t, media + margem_erro_t)

# iii.
z_critico = stats.norm.ppf(0.975)
margem_erro_normal_aproximada = z_critico * desv_padrao / math.sqrt(tamanho)
IC_normal_aproximado = (
    media - margem_erro_normal_aproximada,
    media + margem_erro_normal_aproximada
)

resultados = {
    "n": tamanho,
    "sample_mean": media,
    "sample_sd": desv_padrao,
    "ci95_t": IC,
    "ci95_z": IC_normal_aproximado,
    "width_t": IC[1] - IC[0],
    "width_z": IC_normal_aproximado[1] - IC_normal_aproximado[0]
}

output_file = Path("./resultados/q5_resultados.json")
output_file.parent.mkdir(exist_ok=True, parents=True)
output_file.write_text(
    json.dumps(resultados, indent=4),
)
