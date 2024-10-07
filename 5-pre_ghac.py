import json
import os


raw_path = "input/"

# Carregar o JSON principal
json_file = raw_path + 'full_dataset.json'  # Substitua pelo nome do seu arquivo JSON
with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Criar um diretório para armazenar os JSONs de cada autor
output_dir = raw_path + 'autores_json/'
os.makedirs(output_dir, exist_ok=True)

# Separar os dados por autor e salvar em JSONs individuais
for entry in data:
    author = entry['author']
    author_filename = author.replace(' ', '_') + '.json'
    json_output = os.path.join(output_dir, author_filename)

    # Se o arquivo já existe, adiciona a entrada ao arquivo existente
    if os.path.isfile(json_output):
        with open(json_output, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)

        existing_data.append(entry)

        with open(json_output, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)
    else:
        with open(json_output, 'w', encoding='utf-8') as file:
            json.dump([entry], file, indent=4, ensure_ascii=False)

print("JSONs criados para cada nome ambiguo de autor.")
