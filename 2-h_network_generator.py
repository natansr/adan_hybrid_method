import os
import pickle
import networkx as nx

class HeterogeneousNetworkGenerator:
    def __init__(self):
        self.G = nx.Graph()  # Rede NetworkX

    def read_data(self, dirpath):
        # Carregar e processar arquivos de relacionamentos

        # Documentos e Palavras
        with open(os.path.join(dirpath, "paper_word.txt")) as file:
            for line in file:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    paper_id, word = toks
                    self.G.add_node(paper_id, type='paper')
                    self.G.add_node(word, type='word')
                    self.G.add_edge(paper_id, word, relationship='contains')

        # Documentos e Autores
        with open(os.path.join(dirpath, "paper_author.txt")) as file:
            for line in file:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    paper_id, author_id = toks
                    self.G.add_node(author_id, type='author')
                    self.G.add_edge(paper_id, author_id, relationship='written_by')

        # Documentos e Conferências
        with open(os.path.join(dirpath, "paper_conf.txt")) as file:
            for line in file:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    paper_id, conf_id = toks
                    self.G.add_node(conf_id, type='conference')
                    self.G.add_edge(paper_id, conf_id, relationship='presented_at')

        # Documentos e Organizações
        with open(os.path.join(dirpath, "paper_organization.txt")) as file:
            for line in file:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    paper_id, org_id = toks
                    self.G.add_node(org_id, type='organization')
                    self.G.add_edge(paper_id, org_id, relationship='affiliated_with')

        # Documentos e Títulos
        with open(os.path.join(dirpath, "paper_title.txt")) as file:
            for line in file:
                toks = line.strip().split("\t", 1)
                if len(toks) == 2:
                    paper_id, title = toks
                    title_node_id = f'title_{paper_id}'
                    self.G.add_node(title_node_id, type='title', content=title)
                    self.G.add_edge(paper_id, title_node_id, relationship='has_title')

        # Documentos e Emails
        with open(os.path.join(dirpath, "paper_email.txt")) as file:
            for line in file:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    paper_id, email = toks
                    email_node_id = f'email_{paper_id}'
                    self.G.add_node(email_node_id, type='email', content=email)
                    self.G.add_edge(paper_id, email_node_id, relationship='has_email')

    def save_network(self, filepath):
        # Salvar o grafo para uso futuro
        with open(filepath, 'wb') as file:
            pickle.dump(self.G, file)
        print("Rede heterogênea salva com sucesso.")

def main():
    dirpath = "data/"
    filepath = os.path.join(dirpath, "HeterogeneousNetwork.pkl")

    hng = HeterogeneousNetworkGenerator()
    hng.read_data(dirpath)
    hng.save_network(filepath)

if __name__ == "__main__":
    main()
