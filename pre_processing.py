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
            texts, labels, coauthors, jconfs = process_xml(file_path, idx)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_coauthors.extend(coauthors)
            all_jconfs.extend(jconfs)
            all_xml_ids.extend([idx] * len(texts))  # Repetir o ID para cada publicação no XML
    return all_texts, all_labels, all_coauthors, all_jconfs, all_xml_ids

def process_xml(xml_file, xml_id):
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_data = file.read()
        xml_data = xml_data.replace("&mdash;", "-")
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
        jconf_str = f", Conferência: {jconf}" if jconf is not None else ""
        text = f"{title} (Coautores: {coauthor_str}{jconf_str})"
        texts.append(text)
        labeled_text = f"{xml_id}_{label}"  # Label no novo formato: ID_XML_label_publicacao
        labels.append(labeled_text)
        coauthors.append(coauthor_str)
        jconfs.append(jconf)
    return texts, labels, coauthors, jconfs

xml_dir = "dataset"
csv_dir = "csv"

texts, labels, coauthors, jconfs, xml_ids = process_xmls(xml_dir)

# Divisão dos dados em treinamento e teste
train_texts, val_texts, train_labels, val_labels, train_coauthors, val_coauthors, train_jconfs, val_jconfs, train_xml_ids, val_xml_ids = train_test_split(
    texts, labels, coauthors, jconfs, xml_ids, test_size=0.2, random_state=42
)

# Criar DataFrames para treinamento e validação
train_df = pd.DataFrame({"xml_id": train_xml_ids, "text": train_texts, "label": train_labels, "coauthors": train_coauthors, "jconf": train_jconfs})
val_df = pd.DataFrame({"xml_id": val_xml_ids, "text": val_texts, "label": val_labels, "coauthors": val_coauthors, "jconf": val_jconfs})

# Salvar os DataFrames em arquivos CSV
train_csv_file = os.path.join(csv_dir, "train_data.csv")
val_csv_file = os.path.join(csv_dir, "val_data.csv")
train_df.to_csv(train_csv_file, index=False)
val_df.to_csv(val_csv_file, index=False)

print(f"Dados de treinamento salvos em '{train_csv_file}'")
print(f"Dados de validação salvos em '{val_csv_file}'")
