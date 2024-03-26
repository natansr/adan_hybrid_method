import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split

def process_xmls(xml_dir):
    all_texts = []
    all_labels = []
    all_coauthors = []
    all_jconfs = []
    all_xml_ids = []
    for idx, file_name in enumerate(os.listdir(xml_dir)):
        if file_name.endswith('.xml'):
            file_path = os.path.join(xml_dir, file_name)
            print("Nome do arquivo: " + file_path)
            texts, labels, coauthors, jconfs = process_xml(file_path, idx)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_coauthors.extend(coauthors)
            all_jconfs.extend(jconfs)
            all_xml_ids.extend([idx] * len(texts))  # Repetir o ID para cada publicação no XML
    return all_texts, all_labels, all_coauthors, all_jconfs, all_xml_ids

def process_xml(xml_file, xml_id):
    with open(xml_file, 'r', encoding='utf-8') as file:
        
        
        #tratando os caracteres especiais
        xml_data = file.read()
        xml_data = xml_data.replace("&mdash;", "-")
        xml_data = xml_data.replace("&", "and")
        
        
        
        tree = ET.ElementTree(ET.fromstring(xml_data))
    texts = []
    labels = []
    coauthors = []
    jconfs = []
    for publication in tree.findall(".//publication"):
        title = publication.find("title").text
        label = int(publication.find("label").text)
        coauthor_list = publication.find("authors").text.split(",") if publication.find("authors") is not None else []
        coauthor_str = ", ".join(coauthor_list)
        jconf = publication.find("jconf").text.strip() if publication.find("jconf") is not None else None
        jconf_str = f", Conference: {jconf}" if jconf is not None else ""
        text = f"{title} (Co-authors: {coauthor_str}{jconf_str})"
        texts.append(text)
        labeled_text = f"{xml_id}{label}"  # Label no novo formato: ID_XML_label_publicacao
        labels.append(labeled_text)
        coauthors.append(coauthor_str)
        jconfs.append(jconf)
    return texts, labels, coauthors, jconfs

xml_dir = "dataset"
csv_dir = "csv"

texts, labels, coauthors, jconfs, xml_ids = process_xmls(xml_dir)

# Criar DataFrame
df = pd.DataFrame({"text": texts, "label": labels, "coauthors": coauthors, "jconf": jconfs})

df = pd.DataFrame({"text": texts, "label": labels})

# Salvar o DataFrame em arquivo CSV
csv_file = os.path.join(csv_dir, "data.csv")
df.to_csv(csv_file, index=False, sep=';')

print(f"Dados salvos em '{csv_file}'")



####Gerando dados CSV tb


import csv
import json

# Definir o caminho do arquivo CSV de entrada e o caminho do arquivo JSON de saída
json_file = "json/dataset_cuda.json"

# Lista para armazenar os dados convertidos para JSON
cuda_data = []

# Ler o arquivo CSV e converter para JSON
with open(csv_file, "r") as file:
    csv_reader = csv.reader(file, delimiter=";")
    next(csv_reader)  # Pular o cabeçalho se houver
    for row in csv_reader:
        text, label = row
        cuda_instance = {
            "text": text,
            "label": label
        }
        cuda_data.append(cuda_instance)

# Salvar os dados JSON em um arquivo
with open(json_file, "w") as file:
    json.dump(cuda_data, file, indent=4)

print("Dados convertidos para JSON e salvos com sucesso em", json_file)
