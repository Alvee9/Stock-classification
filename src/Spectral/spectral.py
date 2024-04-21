# import numpy as np
# from scipy.spatial.kdtree import distance_matrix
# from sklearn import metrics
# from utils.dataset import Dataset
# from sklearn.utils.validation import check_symmetric

# dataset = Dataset()

# companies = dataset.read_company_names()

# data_closing = dataset.get_closing_prices(companies, 200)
# data_change = dataset.calc_price_changes(companies, 200)
# data_closing = dataset.normalize(data_closing)
# data_change = np.array(data_change)[:, 0:300]
# data_closing = np.array(data_closing)[:, 0:300]

# def cust_dist(a, b):
#     import math
#     d = 0.0
#     for i in range(1, min(len(a), len(b))):
#         if ((a[i] - a[i - 1]) * (b[i] - b[i - 1]) < 0):
#             d += abs(a[i] - b[i]) * 4
#         else:
#             change_diff = abs((a[i] - a[i - 1]) - (b[i] - b[i - 1]))
#             d += change_diff
#         if i >= 7:
#             if ((a[i] - a[i - 7]) * (b[i] - b[i - 7]) < 0):
#                 d += abs(a[i] - a[i - 7] - (b[i] - b[i - 7])) * 8
#             else:
#                 change_diff = abs((a[i] - a[i - 7]) - (b[i] - b[i - 7]))
#                 d += change_diff * 2
#         if i >= 30:
#             if ((a[i] - a[i - 30]) * (b[i] - b[i - 30]) < 0):
#                 d += abs(a[i] - a[i - 30] - (b[i] - b[i - 30])) * 12
#             else:
#                 change_diff = abs((a[i] - a[i - 30]) - (b[i] - b[i - 30]))
#                 d += change_diff * 3

#         d += abs(a[i] - b[i])/2
#     d = 1 / (1 + d)
#     if math.isnan(d) or math.isinf(d):
#         raise Exception
#     return d


# def get_dist_matrix(data):
#     distance_matrix = []
#     for i in range(len(data)):
#         row = []
#         for j in range(len(data)):
#             row.append(cust_dist(data[i][1:], data[j][1:]))
#         distance_matrix.append(row)
#     # print(distance_matrix)
#     is_sym = check_symmetric(np.array(distance_matrix), raise_exception=True)
#     return distance_matrix


# def clustering(data_closing, data_change, distance_matrix, number_of_clusters=5):
#     from sklearn.cluster import SpectralClustering
#     from matplotlib import pyplot as plt
#     from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#     spt = SpectralClustering(n_clusters=number_of_clusters, affinity='precomputed')
#     labels = spt.fit(distance_matrix).labels_

#     sil_score = silhouette_score(data_closing, labels)
#     dbi = davies_bouldin_score(np.array(data_closing)[:, 1:], labels)
#     chs = calinski_harabasz_score(np.array(data_closing)[:, 1:], labels)
#     print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)

#     for k in range(0, number_of_clusters):
#         plt.clf()
#         clusterSize = 0
#         for idx, r in enumerate(data_closing):
#             if labels[idx] == k:
#                 plt.plot(r[1:], label = companies[int(r[0])])
#                 clusterSize += 1
#         plt.title('Cluster number {} size = {}'.format(k, clusterSize))
#         plt.legend()
#         # plt.show()
#         plt.savefig('../Plots/Spectral/{}.png'.format(k), dpi=600)
    

# distance_matrix = get_dist_matrix(data_closing)

# for i in range(30, 31):
#     print(i)
#     clustering(data_closing, data_change, distance_matrix, i)




import numpy as np
from utils.dataset import Dataset
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from matplotlib import pyplot as plt
from utils.utils import get_edge_matrix

class Spectral:
    def __init__(self) -> None:
        self.dataset = Dataset()
        self.companies = self.dataset.read_company_names()
        self.number_of_clusters = 2
        self.data_closing = []
        self.data_change = []
        self.cluster_labels = []
        self.cluster_sizes = []
        self.clusters_list = []
        self.distance_matrix = [[]]

    def load_data(self):
        self.companies = self.dataset.read_company_names()
        self.data_closing = self.dataset.get_closing_prices(self.companies, 200)
        self.data_change = self.dataset.calc_price_changes(self.companies, 200)
        self.data_closing = self.dataset.normalize(self.data_closing)
        self.data_change = np.array(self.data_change)[:, 0:300]
        self.data_closing = np.array(self.data_closing)[:, 0:300]
        self.distance_matrix = get_edge_matrix(self.data_closing)
    

    def clustering(self, number_of_clusters, distance=None):
        self.number_of_clusters = number_of_clusters
        if distance is None:
            spectral = SpectralClustering(n_clusters=self.number_of_clusters, affinity='nearest_neighbors').fit(np.array(self.data_closing)[:, 1:])
        else:
            spectral = SpectralClustering(n_clusters=self.number_of_clusters, affinity='precomputed').fit(self.distance_matrix)
        self.cluster_labels = list(spectral.labels_)

        self.cluster_sizes = []
        self.cluster_lists = []
        for k in range(0, self.number_of_clusters):
            clusterSize = 0
            cluster_list = []
            for idx, r in enumerate(self.data_closing):
                if self.cluster_labels[idx] == k:
                    clusterSize += 1
                    cluster_list.append(idx)
            self.cluster_sizes.append(clusterSize)
            self.cluster_lists.append(cluster_list)
        

    def get_validation_scores(self):
        sil_score = silhouette_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        dbi = davies_bouldin_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        chs = calinski_harabasz_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        # print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)


        # return ([sil_score, dbi, chs], silhouette_samples(np.array(self.data_closing)[:, 1:], self.cluster_labels))
        return sil_score


    def plot(self, path='../Plots/Spectral/{}.png', save=False):
        for k in range(0, self.number_of_clusters):
            plt.clf()
            clusterSize = 0
            for idx, r in enumerate(self.data_closing):
                if self.cluster_labels[idx] == k:
                    plt.plot(r[1:], label = self.companies[int(r[0])])
                    clusterSize += 1
            plt.title('Cluster number {} size = {}'.format(k, clusterSize))
            plt.legend()
            if save:
                plt.savefig(path.format(k), dpi=600)
            else:
                plt.show()

if __name__ == "__main__":
    spectral = Spectral()
    spectral.load_data()
    spectral.clustering(20)

    spectral.plot('../Plots/Spectral/{}.png', save=True)

# validation = spectral.get_validation_scores()

# sil_avg = []

# for k in range(0, spectral.number_of_clusters):
#     sil_sum = 0
#     for i in spectral.cluster_lists[k]:
#         sil_sum += validation[1][i]
#     sil_avg.append(sil_sum / spectral.cluster_sizes[k])


# for i in range(0, spectral.number_of_clusters):
#     if spectral.cluster_sizes[i] > 7:
#         print("original cluster", i)
#         data = []
#         for j in spectral.cluster_lists[i]:
#             data.append(spectral.data_closing[j])
#         div_spectral = Spectral()
#         div_spectral.data_closing = data
#         div_spectral.distance_matrix = get_edge_matrix(data)
#         div_spectral.clustering(int(np.ceil(spectral.cluster_sizes[i]/7.0)))
#         div_spectral.plot(save=False)



# from K_means.k_means import K_means

# for i in range(0, spectral.number_of_clusters):
#     if spectral.cluster_sizes[i] > 7:
#         print("original cluster", i)
#         data = []
#         for j in spectral.cluster_lists[i]:
#             data.append(spectral.data_closing[j])
#         div_spectral = K_means()
#         div_spectral.data_closing = data
#         # div_spectral.distance_matrix = get_edge_matrix(data)
#         div_spectral.clustering(int(np.ceil(spectral.cluster_sizes[i]/7.0)))
#         div_spectral.plot(save=False)



# spectral = Spectral()
# spectral.load_data()

# for k in range(3, 31):
#     spectral.clustering(k, distance='custom')
#     print(k, spectral.get_validation_scores())