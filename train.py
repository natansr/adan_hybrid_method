import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Carregar dados do CSV
csv_file = "csv/data.csv"
df = pd.read_csv(csv_file, sep=';')

# Dividir dados em treinamento e validação
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizar os dados
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
train_encodings = tokenizer(list(train_df["text"]), truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(list(val_df["text"]), truncation=True, padding=True, return_tensors='pt')

# Configurar o treinamento
training_args = TrainingArguments(
    output_dir="./output",  # Diretório onde os resultados do treinamento serão salvos
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    disable_tqdm=False,
)

# Inicializar o modelo
model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)

# Treinar o modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings,
)

trainer.train()

# Salvar o modelo treinado
output_dir = "./model"
trainer.save_model(output_dir)
