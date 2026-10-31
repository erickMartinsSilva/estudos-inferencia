import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

treatment_df = pd.read_csv("./data/treatment.csv")
treatment_crosstab = pd.crosstab(
    treatment_df["treatment"], treatment_df["improvement"], margins=False
)
treatment_crosstab_dict = treatment_crosstab.to_dict(orient="index")
observed_table = [
    [treatment_crosstab_dict[key]["improved"], treatment_crosstab_dict[key]["not-improved"]]
    for key in treatment_crosstab_dict
]

output_path = Path("./resultados/q5_contingencia.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
table_labels = treatment_crosstab.rename(
    index={"not-treated": "Não tratado", "treated": "Tratado"},
    columns={"improved": "Melhorou", "not-improved": "Não melhorou"}
)
fig, axis = plt.subplots(figsize=(8, 3), layout="constrained")
axis.set_title("Tabela de contingência: tratamento e melhora")
axis.axis("off")
table = axis.table(
    cellText=table_labels.values,
    rowLabels=table_labels.index,
    colLabels=table_labels.columns,
    cellLoc="center",
    bbox=[0.2, 0.15, 0.8, 0.65]
)
table.auto_set_font_size(False)
table.set_fontsize(12)
fig.savefig(output_path, dpi=150)
plt.close(fig)

alpha = 0.05
# h0: tratamento e melhora são independentes
# hA: tratamento e melhora dependem uma da outra

chi2_statistic, pvalue, df, expected_table = chi2_contingency(observed_table, correction=False)

resultados = {
    "observed_table": observed_table,
    "expected_table": expected_table.tolist(),
    "chi2_statistic": chi2_statistic,
    "df": df,
    "pvalue": pvalue,
    "alpha": alpha,
    "decision": "Rejeitar h0" if pvalue <= alpha else "Nao rejeitar h0",
}

output_path = Path("./resultados/q5_resultados.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(json.dumps(resultados, indent=4).encode("utf-8"))
