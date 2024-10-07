import pandas as pd
import random
import json
import numpy as np

# Ler os dados do arquivo CSV sem cabeçalho
data = pd.read_csv('repositories/CustAND/input/full_custand.csv', delimiter='\t', header=None)

# Preencher valores NaN ou "none" nos campos vazios com uma string vazia
data = data.replace({np.nan: '', 'none': ''})

# Criar um dicionário para mapear os rótulos para inteiros
label_mapping = {}
next_label_id = 0

# Função para mapear os rótulos para inteiros
def map_label_to_int(label):
    global next_label_id
    if label not in label_mapping:
        label_mapping[label] = next_label_id
        next_label_id += 1
    return label_mapping[label]

# Criar um dataset vazio
dataset = []

# Iterar sobre cada linha dos dados
for index, row in data.iterrows():
    label = map_label_to_int(row[1])
    author = row[3]
    id = row[2]
    print(row[2])
    organization = row[5]
    email = row[12]
    conf = row[10]
    abstract = row[13]
    coauthors = row[8].split(';') if row[8] else []
    coauthors = [coauthor.strip() for coauthor in coauthors if coauthor.strip()]  # Remover espaços em branco e coautores vazios
    title = row[9]
    keywords = row[14]  # Ler o campo "keywords"

    # Criar o dicionário de dados
    data_entry = {
        'id': id,
        'label': label,
        'author': author,
        'title': title,
        'organization': organization,
        'abstract': abstract,
        'email': email, 
        'conf': conf,
        'coauthors': coauthors,
        'keywords': keywords  # Incluir o campo "keywords"
    }

    # Adicionar os dados ao dataset
    dataset.append(data_entry)

# Embaralhar o dataset
random.shuffle(dataset)

# Dividir o dataset em treinamento (80%), teste (10%) e validação (10%)
train_size = int(0.9 * len(dataset))
test_size = int(0.1 * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:train_size+test_size]
validation_data = dataset[train_size+test_size:]


'''
# Exportar o conjunto de treinamento para um arquivo JSON com quebras de linha
with open('train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4, separators=(',', ': '))

# Exportar o conjunto de teste para um arquivo JSON com quebras de linha
with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4, separators=(',', ': '))

# Exportar o conjunto de validação para um arquivo JSON com quebras de linha
with open('validation_data.json', 'w') as f:
    json.dump(validation_data, f, indent=4, separators=(',', ': '))

'''

with open('repositories/CustAND/input/full_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4, separators=(',', ': '))


# Imprimir a contagem total de rótulos
total_labels = len(label_mapping)
print("Total de rótulos:", total_labels)
