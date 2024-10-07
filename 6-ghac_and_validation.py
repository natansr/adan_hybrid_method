import numpy as np
import pandas as pd
import pickle
import os
import json
import time
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from scipy.sparse.csgraph import connected_components

# Diretórios e arquivos
dirpath = ""
data_dir = dirpath + 'data/'

# Função para carregar embeddings da GCN
def load_gcn_embeddings(embedding_type):
    if embedding_type == "tfidf":
        with open(dirpath + 'embeddings/pemb_final_gcn_gcn_with_tf-idf_embeddings.pkl', "rb") as file_obj:
            return pickle.load(file_obj)
    elif embedding_type == "word2vec":
        with open(dirpath + 'embeddings/pemb_final_gcn_gcn_with_word2vec_embeddings.pkl', "rb") as file_obj:
            return pickle.load(file_obj)
    elif embedding_type == "scibert":
        with open(dirpath + 'embeddings/pemb_final_gcn_gcn_with_scibert_embeddings.pkl', "rb") as file_obj:
            return pickle.load(file_obj)

# Função BCubed
def bcubed_precision_recall(correct_labels, predicted_labels):
    precision = []
    recall = []

    for i in range(len(correct_labels)):
        pred_cluster = predicted_labels[i]
        true_cluster = correct_labels[i]

        pred_cluster_indices = np.where(predicted_labels == pred_cluster)[0]
        true_cluster_indices = np.where(correct_labels == true_cluster)[0]

        intersection = len(np.intersect1d(pred_cluster_indices, true_cluster_indices))
        precision.append(intersection / len(pred_cluster_indices))
        recall.append(intersection / len(true_cluster_indices))

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)

    if mean_precision + mean_recall == 0:
        f1 = 0
    else:
        f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

    return mean_precision, mean_recall, f1

# Funções de Avaliação
def pairwise_evaluate(correct_labels, pred_labels):
    if len(correct_labels) != len(pred_labels):
        print("As listas têm tamanhos diferentes. Não é possível realizar a comparação.")
        return 0, 0, 0

    TP = 0.0  # Pairs Correctly Predicted To Same Author
    TP_FP = 0.0  # Total Pairs Predicted To Same Author
    TP_FN = 0.0  # Total Pairs To Same Author

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1

def calculate_ACP_AAP(correct_labels, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    ACP = 0.0
    AAP = 0.0

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_author_labels = correct_labels[cluster_indices]
        unique_author_labels, author_counts = np.unique(cluster_author_labels, return_counts=True)

        max_count = np.max(author_counts)
        ACP += max_count / len(cluster_indices)

        if len(unique_author_labels) > 1:
            min_count = np.min(author_counts)
            AAP += 1 - (min_count / len(cluster_indices))
        else:
            AAP += 1

    ACP /= len(unique_clusters)
    AAP /= len(unique_clusters)

    return ACP, AAP

def calculate_KMetric(ACP, AAP):
    return np.sqrt(ACP * AAP)

# Função GHAC
def GHAC(mlist, n_clusters=-1):
    distance = []

    for i in range(len(mlist)):
        gtmp = []
        for j in range(len(mlist)):
            if i < j:
                cosdis = np.dot(mlist[i], mlist[j]) / (np.linalg.norm(mlist[i]) * (np.linalg.norm(mlist[j])))
                gtmp.append(cosdis)
            elif i > j:
                gtmp.append(distance[j][i])
            else:
                gtmp.append(0)
        distance.append(gtmp)

    distance = np.array(distance)
    distance = np.multiply(distance, -1)

    if n_clusters == -1:
        best_m = -10000000
        n_components1, labels = connected_components(distance)

        distance[distance <= 0.5] = 0
        G = nx.from_numpy_matrix(distance)
        n_components, labels = connected_components(distance)

        for k in range(n_components, n_components1 - 1, -1):
            model_HAC = AgglomerativeClustering(linkage="average", metric='precomputed', n_clusters=k)
            model_HAC.fit(distance)
            labels = model_HAC.labels_

            part = {}
            for j in range(len(labels)):
                part[j] = labels[j]

            mod = nx.algorithms.community.quality.modularity(G, [set(np.where(np.array(labels) == i)[0]) for i in range(len(set(labels)))])
            if mod > best_m:
                best_m = mod
                best_labels = labels
        labels = best_labels
    else:
        model_HAC = AgglomerativeClustering(linkage='average', metric='cosine', n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_

    return labels

def adjust_list_sizes(correct_labels, predicted_labels):
    len_correct = len(correct_labels)
    len_predicted = len(predicted_labels)

    if len_correct > len_predicted:
        # Adicionar valores arbitrários (-1) à lista de rótulos previstos
        predicted_labels = np.pad(predicted_labels, (0, len_correct - len_predicted), constant_values=-1)
    elif len_predicted > len_correct:
        # Adicionar valores arbitrários (-1) à lista de rótulos corretos
        correct_labels = np.pad(correct_labels, (0, len_predicted - len_correct), constant_values=-1)

    return correct_labels, predicted_labels

def cluster_evaluate(method, embedding_type):
    path = dirpath + 'input/autores_json/'
    all_authors = set()

    for fname in os.listdir(path):
        with open(path + fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for entry in data:
                all_authors.add(entry['author'])

    ktrue = []
    kpre = []
    all_pairwise_precision = []
    all_pairwise_recall = []
    all_pairwise_f1 = []
    all_AAP = []
    all_ACP = []
    all_KMetric = []
    all_bcubed_precision = []
    all_bcubed_recall = []
    all_bcubed_f1 = []

    results = []

    for author in all_authors:
        fname = author.replace(" ", "_") + ".json"
        with open(path + fname, 'r', encoding='utf-8') as file:
            data = json.load(file)

        correct_labels = [entry['label'] for entry in data]

        if len(correct_labels) < 2:
            continue

        papers = ['i' + str(entry['id']) for entry in data]
        mlist = [new_pembd[pid] for pid in papers if pid in new_pembd]

        if len(mlist) == 0:
            continue

        t0 = time.time()

        if method == "GHAC_nok":
            labels = GHAC(mlist)
        elif method == "GHAC":
            labels = GHAC(mlist, len(set(correct_labels)))

        time1 = time.time() - t0

        # Ajustar tamanhos das listas se forem diferentes
        correct_labels, labels = adjust_list_sizes(np.array(correct_labels), np.array(labels))

        #print("Author:" + author)
        #print("Correct labels:", correct_labels)
        #print("Predicted labels:", labels)

        # Comparação com listas ajustadas
        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, labels)
        ACP, AAP = calculate_ACP_AAP(correct_labels, labels)
        K = calculate_KMetric(ACP, AAP)

        # Avaliação BCubed
        bcubed_precision, bcubed_recall, bcubed_f1 = bcubed_precision_recall(correct_labels, labels)

        results.append([author, pairwise_precision, pairwise_recall, pairwise_f1, ACP, AAP, K, bcubed_precision, bcubed_recall, bcubed_f1])

        ktrue.append(len(set(correct_labels)))
        kpre.append(len(set(labels)))

        all_pairwise_precision.append(pairwise_precision)
        all_pairwise_recall.append(pairwise_recall)
        all_pairwise_f1.append(pairwise_f1)
        all_AAP.append(AAP)
        all_ACP.append(ACP)
        all_KMetric.append(K)
        all_bcubed_precision.append(bcubed_precision)
        all_bcubed_recall.append(bcubed_recall)
        all_bcubed_f1.append(bcubed_f1)

    avg_pairwise_precision = np.mean(all_pairwise_precision)
    avg_pairwise_recall = np.mean(all_pairwise_recall)
    avg_pairwise_f1 = np.mean(all_pairwise_f1)
    avg_AAP = np.mean(all_AAP)
    avg_ACP = np.mean(all_ACP)
    avg_KMetric = np.mean(all_KMetric)
    avg_bcubed_precision = np.mean(all_bcubed_precision)
    avg_bcubed_recall = np.mean(all_bcubed_recall)
    avg_bcubed_f1 = np.mean(all_bcubed_f1)

    results.append(["Average", avg_pairwise_precision, avg_pairwise_recall, avg_pairwise_f1, avg_ACP, avg_AAP, avg_KMetric, avg_bcubed_precision, avg_bcubed_recall, avg_bcubed_f1])

    results_df = pd.DataFrame(results, columns=["Author", "Pairwise Precision", "Pairwise Recall", "Pairwise F1", "ACP", "AAP", "K Metric", "BCubed Precision", "BCubed Recall", "BCubed F1"])
    results_df.to_csv(dirpath + f'results/clustering_results_{embedding_type}.csv', index=False)

    print("+--------------------------------------------------------")
    print("|Cluster method:", method)
    print("|Embeddings:", embedding_type)
    print("|Macro-F1")
    print("|Precision: ", avg_pairwise_precision)
    print("|Recall: ", avg_pairwise_recall)
    print("|F1", avg_pairwise_f1)
    print("|ACP Avg:", avg_ACP)
    print("|AAP Avg:", avg_AAP)
    print("|K Metric Avg:", avg_KMetric)
    print("|BCubed Precision: ", avg_bcubed_precision)
    print("|BCubed Recall: ", avg_bcubed_recall)
    print("|BCubed F1: ", avg_bcubed_f1)
    print("+--------------------------------------------------------")


def main():
    #for embedding_type in ["tfidf", "word2vec", "scibert"]:
    for embedding_type in ["scibert"]:
        global new_pembd
        new_pembd = load_gcn_embeddings(embedding_type)
        cluster_evaluate('GHAC', embedding_type)

if __name__ == "__main__":
    main()
