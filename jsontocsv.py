import json
import pandas as pd

# Cargar el JSON
with open('testset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraer solo las muestras aprobadas y la info relevante
records = []
for sample in data:
    if sample.get("approval_status") == "approved":
        eval_sample = sample["eval_sample"]
        records.append({
            "user_input": eval_sample["user_input"],
            "reference_contexts": eval_sample["reference_contexts"],
            "reference": eval_sample["reference"],
            # Añade más campos si el evaluador los necesita
        })

# Convertir a DataFrame
df = pd.DataFrame(records)

# Guardar como CSV
df.to_csv("testset_adaptado.csv", index=False)
print("✅ testset_adaptado.csv creado correctamente SOLO con aprobados")
