import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Diretório onde os dados estão armazenados
data_dir = "csv"

# Ler o arquivo CSV contendo os dados
df = pd.read_csv(os.path.join(data_dir, "data.csv"), sep=";")

# Visualizar os primeiros registros do DataFrame
print(df.head())

# Lista das colunas que serão utilizadas para a classificação multirrótulo
columns = ['text', 'label']
meta_dfs = df[columns]

# Converter valores da coluna 'label' para strings
meta_dfs['label'] = meta_dfs['label'].astype(str)

# Dividir as categorias em uma lista de strings
meta_dfs['label'] = meta_dfs.label.apply(lambda x: x.split(','))

# Encontrar a quantidade de categorias distintas
distinct_labels = set()
for labels in meta_dfs['label']:
    distinct_labels.update(labels)

# Codificar as categorias em colunas binárias
mlb = MultiLabelBinarizer(classes=list(distinct_labels))
labels = mlb.fit_transform(meta_dfs['label'])

# Criar um novo DataFrame com as colunas 'text' e as colunas binárias de labels
df = pd.concat([meta_dfs[['text']], pd.DataFrame(labels, columns=mlb.classes_)], axis=1)

# Salvar o DataFrame preprocessado em um arquivo CSV
df.to_csv(os.path.join(data_dir, "seu_dataset_preproc.csv"), sep="\t", header=True, index=False)

# Obter a lista de classes de autor
categories = mlb.classes_

# Imprimir a quantidade de categorias criadas
print("Quantidade de categorias:", len(categories))
print("Categorias:", categories)

# Subamostragem aleatória para reduzir o tamanho dos dados para visualização
sample_pca = df.sample(n=1000, random_state=42)

# Vetorização TF-IDF dos resumos
X = TfidfVectorizer().fit_transform(sample_pca.text).toarray()

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
data2D = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])

# Adicionar as categorias correspondentes aos resumos
data2D['category'] = sample_pca.iloc[:,1:].idxmax(axis=1)

# Plotar o gráfico de dispersão
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data2D, x='PC1', y='PC2', hue='category')
plt.title("PCA dos Resumos Vetorizados", fontsize=18, fontweight='bold')
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)
plt.legend(title="Categoria")
plt.show()
