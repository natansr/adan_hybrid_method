import os
import pandas as pd

import pickle
from gensim.models import word2vec
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import re
import os
import numpy as np


import os
import json
import re
import pickle

data_dir = ""
input_json_file = 'input/full_dataset.json'

# Carregar o JSON
with open(os.path.join(data_dir, input_json_file), 'r') as file:
    data = json.load(file)

# Dicionários para mapear IDs
author_id_dict = {}
paper_id_dict = {}  # Dicionário para mapear IDs de documento
word_id_dict = {}  # Dicionário para mapear IDs de palavras
conf_id_dict = {}  # Dicionário para mapear IDs de conferências
organization_id_dict = {}  # Dicionário para mapear IDs de organizações

# Função para gerar ID do autor e transformar em inteiro
def generate_author_id(author_name, label):
    author_id = f"{author_name}_{label}"  # Convertendo o label para string
    if author_id not in author_id_dict:
        author_id_dict[author_id] = len(author_id_dict) + 1  # Começando de 1 para IDs
    return author_id_dict[author_id]

# Função para gerar ID do documento e transformar em inteiro
def generate_paper_id(label):
    if label not in paper_id_dict:
        paper_id_dict[label] = len(paper_id_dict) + 1  # Começando de 1 para IDs
    return paper_id_dict[label]

# Função para gerar ID da palavra e transformar em inteiro
def generate_word_id(word):
    if word not in word_id_dict:
        word_id_dict[word] = len(word_id_dict) + 1  # Começando de 1 para IDs
    return word_id_dict[word]

# Função para gerar ID da conferência e transformar em inteiro
def generate_conf_id(conf_name):
    if conf_name not in conf_id_dict:
        conf_id_dict[conf_name] = len(conf_id_dict) + 1  # Começando de 1 para IDs
    return conf_id_dict[conf_name]

# Função para gerar ID da organização e transformar em inteiro
def generate_organization_id(org_name):
    if org_name not in organization_id_dict:
        organization_id_dict[org_name] = len(organization_id_dict) + 1  # Começando de 1 para IDs
    return organization_id_dict[org_name]

# Inicializar arquivos de texto para armazenar dados
f1 = open(os.path.join(data_dir, 'data/paper_author.txt'), 'w', encoding='utf-8')
f2 = open(os.path.join(data_dir, 'data/paper_conf.txt'), 'w', encoding='utf-8')  # Novo arquivo para conferências (ID)
f3 = open(os.path.join(data_dir, 'data/paper_word.txt'), 'w', encoding='utf-8')
f4 = open(os.path.join(data_dir, 'data/paper_author1.txt'), 'w', encoding='utf-8')
f5 = open(os.path.join(data_dir, 'data/paper_title.txt'), 'w', encoding='utf-8')
f6 = open(os.path.join(data_dir, 'data/paper_abstract.txt'), 'w', encoding='utf-8')
f7 = open(os.path.join(data_dir, 'data/paper_conf_name.txt'), 'w', encoding='utf-8')  # Novo arquivo para nome da conferência
f8 = open(os.path.join(data_dir, 'data/paper_organization.txt'), 'w', encoding='utf-8')  # Novo arquivo para organização
f9 = open(os.path.join(data_dir, 'data/paper_email.txt'), 'w', encoding='utf-8')  # Novo arquivo para emails
f10 = open(os.path.join(data_dir, 'data/paper_authors_name.txt'), 'w', encoding='utf-8')  # Novo arquivo para nome dos autores

# Processar cada entrada no JSON
for entry in data:
    label = entry['label']
    id = entry['id']
    paper_id = generate_paper_id(id)  # Gerar ID único para o documento

    authors = entry['coauthors']
    email = entry.get('email', '')  # String de email
    title = entry['title']
    abstract = entry['abstract']
    keywords = entry['keywords']
    conference = entry.get('conf', '')  # Nome da conferência
    organization = entry.get('organization', '')  # Nome da organização

    # Gerar IDs para autores e transformar em inteiros
    author_ids_int = []
    author_names = []  # Lista para armazenar os nomes dos autores
    for author in authors:
        author_id_int = generate_author_id(author, label)
        author_ids_int.append(str(author_id_int))
        author_names.append(author)  # Armazenar os nomes dos autores

    # Escrever no arquivo de autores
    for author_id_int in author_ids_int:
        f1.write(f'i{paper_id}\t{author_id_int}\n')  # Associa cada autor ao documento

    # Escrever os nomes dos autores no novo arquivo
    f10.write(f'i{paper_id}\t{", ".join(author_names)}\n')  # Escrever os nomes dos autores

    # Verificar se existem palavras-chave e escrever no arquivo de palavras-chave
    if keywords:
        keyword_list = re.split(r'[;,]', keywords)  # Split por vírgula ou ponto e vírgula
        for keyword in keyword_list:
            keyword = keyword.strip()  # Remover espaços em branco em excesso
            word_id = generate_word_id(keyword)  # Gerar ID único para a palavra
            f3.write(f'i{paper_id}\t{word_id}\n')
    else:
        f3.write(f'i{paper_id}\t\n')  # Se não houver keywords, escrever linha vazia

    # Conferência
    if conference:
        conf_id = generate_conf_id(conference)  # Gerar ID único para a conferência
        f2.write(f'i{paper_id}\t{conf_id}\n')  # Associa o paper ao ID da conferência
        f7.write(f'i{paper_id}\t{conference}\n')  # Nome da conferência

    # Organização
    if organization:
        org_id = generate_organization_id(organization)  # Gerar ID único para a organização
        f8.write(f'i{paper_id}\t{org_id}\n')  # Associa o paper ao ID da organização

    # Email
    if email:
        f9.write(f'i{paper_id}\t{email}\n')  # Associa o email ao paper
    else:
        f9.write(f'i{paper_id}\t\n')  # Se não houver email, escrever linha vazia

    # Escrever nos arquivos de título e abstract
    f4.write(f'i{paper_id}\t{";".join(author_ids_int)}\n')
    f5.write(f'i{paper_id}\t{title}\n')
    f6.write(f'i{paper_id}\t{abstract}\n')

# Fechar os arquivos após a escrita
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()
f9.close()
f10.close()

print(f'Processamento concluído. Total de {len(author_id_dict)} autores, {len(conf_id_dict)} conferências, {len(organization_id_dict)} organizações.')
