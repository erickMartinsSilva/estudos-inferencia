import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats


def intervalo_wald(p_chapeu, n, nivel_confianca):
    z_critico = stats.norm.ppf((1 + nivel_confianca) / 2)
    margem_erro = z_critico * math.sqrt(p_chapeu * (1 - p_chapeu) / n)
    return (
        float(p_chapeu - margem_erro),
        float(p_chapeu + margem_erro),
    )


data = pd.read_csv("./data/survey.csv")

# i.
data_w_hnd = data["W.Hnd"].dropna()
hand_n = len(data_w_hnd)
left_count = int((data_w_hnd == "Left").sum())
right_count = int((data_w_hnd == "Right").sum())
left_proportion = left_count / hand_n

hand_success_condition = bool(hand_n * left_proportion >= 10)
hand_failure_condition = bool(hand_n * (1 - left_proportion) >= 10)

# ii.
ci90_left = intervalo_wald(left_proportion, hand_n, 0.90)
ci95_left = intervalo_wald(left_proportion, hand_n, 0.95)

# iii.
data_sex = data["Sex"].dropna()
sex_n = len(data_sex)
female_count = int((data_sex == "Female").sum())
male_count = int((data_sex == "Male").sum())
female_proportion = female_count / sex_n

sex_success_condition = bool(sex_n * female_proportion >= 10)
sex_failure_condition = bool(sex_n * (1 - female_proportion) >= 10)

ci90_female = intervalo_wald(female_proportion, sex_n, 0.90)
ci95_female = intervalo_wald(female_proportion, sex_n, 0.95)

# iv.
z_critico_90 = stats.norm.ppf((1 + 0.90) / 2)
required_n_sex_90_e02 = math.ceil(
    z_critico_90**2 * 0.25 / 0.02**2
)
sample_sufficient = bool(sex_n >= required_n_sex_90_e02)

resultados = {
    "left_count": left_count,
    "right_count": right_count,
    "hand_n": hand_n,
    "left_proportion": left_proportion,
    "hand_success_condition": hand_success_condition,
    "hand_failure_condition": hand_failure_condition,
    "ci90_left": ci90_left,
    "ci95_left": ci95_left,
    "ci90_female": ci90_female,
    "ci95_female": ci95_female,
    "female_count": female_count,
    "male_count": male_count,
    "sex_n": sex_n,
    "female_proportion": female_proportion,
    "sex_success_condition": sex_success_condition,
    "sex_failure_condition": sex_failure_condition,
    "required_n_sex_90_e02": required_n_sex_90_e02,
    "sample_sufficient": sample_sufficient,
}

output_file = Path("./resultados/q6_resultados.json")
output_file.parent.mkdir(exist_ok=True, parents=True)
output_file.write_text(
    json.dumps(resultados, indent=4),
)
