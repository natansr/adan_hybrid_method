import pandas as pd

# Carregar os dados do CSV
df = pd.read_csv("csv/train_data.csv")

# Contar os valores únicos na coluna de rótulos
num_classes = df["label"].nunique()

print("Número de classes:", num_classes)
