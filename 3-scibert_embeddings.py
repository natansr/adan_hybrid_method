import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import os
from tqdm import tqdm
import numpy as np

# Diretórios e arquivos
dirpath = ""
data_dir = dirpath + 'data/'
model_dir = "/scibert_model/output22"  # Diretório onde está o seu modelo fine-tunado

# Carregar títulos, resumos, autores, organizações, conferências e emails dos artigos
paperid_title = {}
paperid_abstract = {}
paperid_authors = {}
paperid_organizations = {}
paperid_conferences = {}
paperid_emails = {}

# Carregar títulos
with open(os.path.join(data_dir, "paper_title.txt"), encoding='utf-8') as title_file:
    for line in title_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_title[toks[0]] = toks[1]

# Carregar resumos
with open(os.path.join(data_dir, "paper_abstract.txt"), encoding='utf-8') as abstract_file:
    for line in abstract_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_abstract[toks[0]] = toks[1]

# Carregar autores
with open(os.path.join(data_dir, "paper_authors_name.txt"), encoding='utf-8') as author_file:
    for line in author_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_authors[toks[0]] = toks[1]

# Carregar organizações
with open(os.path.join(data_dir, "paper_organization.txt"), encoding='utf-8') as org_file:
    for line in org_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_organizations[toks[0]] = toks[1]

# Carregar conferências
with open(os.path.join(data_dir, "paper_conf_name.txt"), encoding='utf-8') as conf_file:
    for line in conf_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_conferences[toks[0]] = toks[1]

# Carregar emails
with open(os.path.join(data_dir, "paper_email.txt"), encoding='utf-8') as email_file:
    for line in email_file:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            paperid_emails[toks[0]] = toks[1]

# Combine títulos, abstracts, autores, organizações, conferências e emails
documents = []
paper_ids = []
for paperid in paperid_title:
    title = paperid_title[paperid]
    abstract = paperid_abstract.get(paperid, "")
    authors = paperid_authors.get(paperid, "")
    organization = paperid_organizations.get(paperid, "")
    conference = paperid_conferences.get(paperid, "")
    email = paperid_emails.get(paperid, "")

    combined_text = f"{title} {abstract} {authors} {organization} {conference} {email}".strip()
    documents.append(combined_text)
    paper_ids.append(paperid)

# Carregar modelo SciBERT pré-treinado e tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

# Função para obter embeddings usando SciBERT
def get_scibert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Usar a média dos embeddings da última camada
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Extrair embeddings SciBERT para todos os documentos com tqdm
print("Extraindo embeddings com SciBERT...")
paper_vec = {}
for i, doc in enumerate(tqdm(documents, desc="Extraindo Embeddings")):
    paper_vec[paper_ids[i]] = get_scibert_embedding(doc)

# Verificar se todos os embeddings são arrays numpy
for key, value in paper_vec.items():
    if not isinstance(value, np.ndarray):
        print(f"Warning: Embedding for paper ID {key} is not a numpy array")

# Salvar os embeddings
with open(os.path.join(data_dir, 'scibert_emb.pkl'), "wb") as file_obj:
    pickle.dump(paper_vec, file_obj)

print("Extração de embeddings com SciBERT concluída e embeddings salvos.")
