import numpy as np
from scipy.spatial.kdtree import distance_matrix
from sklearn import metrics
from utils.dataset import Dataset
from sklearn.utils.validation import check_symmetric

dataset = Dataset()

companies = dataset.read_company_names()

data_closing = dataset.get_closing_prices(companies, 200)
data_change = dataset.calc_price_changes(companies, 200)
data_closing = dataset.normalize(data_closing)
data_change = np.array(data_change)[:, 0:300]
data_closing = np.array(data_closing)[:, 0:300]

def cust_dist(a, b):
    import math
    d = 0.0
    for i in range(1, min(len(a), len(b))):
        if ((a[i] - a[i - 1]) * (b[i] - b[i - 1]) < 0):
            d += abs(a[i] - b[i]) * 4
        else:
            change_diff = abs((a[i] - a[i - 1]) - (b[i] - b[i - 1]))
            d += change_diff
        if i >= 7:
            if ((a[i] - a[i - 7]) * (b[i] - b[i - 7]) < 0):
                d += abs(a[i] - a[i - 7] - (b[i] - b[i - 7])) * 8
            else:
                change_diff = abs((a[i] - a[i - 7]) - (b[i] - b[i - 7]))
                d += change_diff * 2
        if i >= 30:
            if ((a[i] - a[i - 30]) * (b[i] - b[i - 30]) < 0):
                d += abs(a[i] - a[i - 30] - (b[i] - b[i - 30])) * 12
            else:
                change_diff = abs((a[i] - a[i - 30]) - (b[i] - b[i - 30]))
                d += change_diff * 3

        d += abs(a[i] - b[i])/2
    d = 1 / (1 + d)
    if math.isnan(d) or math.isinf(d):
        raise Exception
    return d


def get_dist_matrix(data):
    distance_matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            row.append(cust_dist(data[i][1:], data[j][1:]))
        distance_matrix.append(row)
    # print(distance_matrix)
    is_sym = check_symmetric(np.array(distance_matrix), raise_exception=True)
    return distance_matrix


def clustering(data_closing, data_change, distance_matrix, number_of_clusters=5):
    from sklearn.cluster import SpectralClustering
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    spt = SpectralClustering(n_clusters=number_of_clusters, affinity='precomputed')
    labels = spt.fit(distance_matrix).labels_

    sil_score = silhouette_score(data_closing, labels)
    dbi = davies_bouldin_score(np.array(data_closing)[:, 1:], labels)
    chs = calinski_harabasz_score(np.array(data_closing)[:, 1:], labels)
    print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)

    for k in range(0, number_of_clusters):
        plt.clf()
        clusterSize = 0
        for idx, r in enumerate(data_closing):
            if labels[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        # plt.show()
        plt.savefig('../Plots/Spectral/{}.png'.format(k), dpi=600)
    

distance_matrix = get_dist_matrix(data_closing)

for i in range(33, 34):
    print(i)
    clustering(data_closing, data_change, distance_matrix, i)

