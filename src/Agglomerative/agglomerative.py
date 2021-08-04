import numpy as np
from scipy.spatial.kdtree import distance_matrix
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data_closing = dataset.get_closing_prices(companies, 50)
data_change = dataset.calc_price_changes(companies, 50)
data_closing = dataset.normalize(data_closing)
data_change = np.array(data_change)[:, 0:100]
data_closing = np.array(data_closing)[:, 0:100]

def cust_dist(a, b):
    d = 0.0
    for i in range(0, min(len(a), len(b))):
        d += abs(a[i] - b[i])
    
    return d


def get_dist_matrix(data):
    distance_matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            row.append(cust_dist(data[i][1:], data[j][1:]))
        distance_matrix.append(row)
    return distance_matrix


def clustering(data_closing, data_change, number_of_clusters=5):
    from sklearn.cluster import AgglomerativeClustering
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score

    distance_matrix = get_dist_matrix(data_closing)

    agl = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='precomputed', linkage='average')
    labels = agl.fit(distance_matrix).labels_

    sil_score = silhouette_score(np.array(data_closing)[:, 1:], labels)
    print(sil_score)

    for k in range(0, number_of_clusters):
        clusterSize = 0
        for idx, r in enumerate(data_closing):
            if labels[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()



for i in range(6, 7):
    print(i)
    clustering(data_closing, data_change, i)

