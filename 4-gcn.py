import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pickle
import os

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

# Diretórios e arquivos
dirpath = ""
data_dir = dirpath + 'data/'

# Carregar a rede heterogênea
with open(os.path.join(data_dir, "HeterogeneousNetwork.pkl"), 'rb') as file:
    G = pickle.load(file)

# Função para carregar embeddings
def load_embeddings(embedding_type):
    if embedding_type == "tfidf":
        with open(data_dir + 'tfidf_emb.pkl', "rb") as file_obj:
            return pickle.load(file_obj)
    elif embedding_type == "scibert":
        with open(data_dir + 'scibert_emb.pkl', "rb") as file_obj:
            return pickle.load(file_obj)
    elif embedding_type == "word2vec":
        with open(data_dir + 'word2vec_emb.pkl', "rb") as file_obj:
            return pickle.load(file_obj)
    elif embedding_type == "random":
        return None

def prepare_features(embedding_type, G, embeddings):
    nodes = list(G.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in G.edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)  # Mover para GPU

    if embedding_type in ["tfidf", "scibert", "word2vec"]:
        sample_embedding = next(iter(embeddings.values()))
        if isinstance(sample_embedding, np.ndarray):
            embedding_dim = sample_embedding.shape[0]
        else:
            raise ValueError(f"Unexpected embedding type: {type(sample_embedding)}")
        features = []
        for node in nodes:
            if node in embeddings and isinstance(embeddings[node], np.ndarray) and embeddings[node].shape == (embedding_dim,):
                features.append(embeddings[node])
            else:
                features.append(np.zeros(embedding_dim))
    else:
        embedding_dim = 128
        features = np.random.normal(loc=0.5, scale=10000.0, size=(len(nodes), embedding_dim))

    features = np.array(features)
    x = torch.tensor(features, dtype=torch.float).to(device)  # Mover para GPU

    return Data(x=x, edge_index=edge_index), nodes, embedding_dim

# Definir o modelo GCN com quantidade de camadas ajustável
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, 512))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(512, 512))

        self.convs.append(GCNConv(512, 512))
        self.fc = torch.nn.Linear(512, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.fc(x)
        return x

def train_gcn(data, input_dim, num_layers, epochs=100):
    model = GCN(input_dim, num_layers).to(device)  # Mover o modelo para GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(epochs):
        loss = train()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        new_embeddings = model(data).cpu().numpy()  # Mover os embeddings de volta para a CPU

    return new_embeddings

def save_embeddings(embeddings, nodes, filename):
    embeddings_dict = {node: embeddings[idx] for idx, node in enumerate(nodes)}
    with open(filename, "wb") as file_obj:
        pickle.dump(embeddings_dict, file_obj)

# Configurações de experimentos
configurations = [
    {"embedding": "scibert", "description": "GCN with SciBERT embeddings", "output_dim": 768, "num_layers": 3},  # Ajustável via 'num_layers'
    #{"embedding": "tfidf", "description": "GCN with TF-IDF embeddings", "output_dim": 128, "num_layers": 3},    # Ajustável via 'num_layers'
    #{"embedding": "word2vec", "description": "GCN with Word2Vec embeddings", "output_dim": 128, "num_layers": 3}  # Ajustável via 'num_layers'
]

# Parâmetros comuns
epochs = 1000

# Rodar experimentos
for config in configurations:
    embeddings = load_embeddings(config['embedding'])
    data, nodes, input_dim = prepare_features(config['embedding'], G, embeddings)
    new_embeddings = train_gcn(data, input_dim=input_dim, num_layers=config["num_layers"], epochs=epochs)

    save_embeddings(new_embeddings, nodes, dirpath + f'embeddings/pemb_final_gcn_{config["description"].replace(" ", "_").lower()}.pkl')

    print(f"{config['description']} completed and embeddings saved.")
