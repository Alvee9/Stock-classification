import numpy as np
from sklearn import metrics
from utils.dataset import Dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from matplotlib import pyplot as plt
from utils.utils import cust_dist, get_dist_matrix

class Agglomerative:
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
        self.distance_matrix = get_dist_matrix(self.data_closing)
    

    def clustering(self, number_of_clusters, distance=None):
        self.number_of_clusters = number_of_clusters
        if distance is None:
            agglo = AgglomerativeClustering(n_clusters=self.number_of_clusters).fit(np.array(self.data_closing)[:, 1:])
        else:
            agglo = AgglomerativeClustering(n_clusters=self.number_of_clusters, affinity='precomputed', linkage='average').fit(self.distance_matrix)
        self.cluster_labels = list(agglo.labels_)
        
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
        sil_score = silhouette_score(self.distance_matrix, self.cluster_labels, metric='precomputed')
        dbi = davies_bouldin_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        chs = calinski_harabasz_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        # print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)


        return ([sil_score, dbi, chs], silhouette_samples(np.array(self.data_closing)[:, 1:], self.cluster_labels))


    def plot(self, path='../Plots/Agglo/{}.png', save=False):
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


# agglo = Agglomerative()
# agglo.load_data()
# agglo.clustering(20)

# agglo.plot('../Plots/Agglo/{}.png', save=True)

# validation = agglo.get_validation_scores()

# sil_avg = []

# for k in range(0, agglo.number_of_clusters):
#     sil_sum = 0
#     for i in agglo.cluster_lists[k]:
#         sil_sum += validation[1][i]
#     sil_avg.append(sil_sum / agglo.cluster_sizes[k])


# for i in range(0, agglo.number_of_clusters):
#     if agglo.cluster_sizes[i] > 7:
#         print("original cluster", i)
#         data = []
#         for j in agglo.cluster_lists[i]:
#             data.append(agglo.data_closing[j])
#         div_agglo = Agglomerative()
#         div_agglo.data_closing = data
#         div_agglo.distance_matrix = get_dist_matrix(data)
#         div_agglo.clustering(int(np.ceil(agglo.cluster_sizes[i]/7.0)))
#         div_agglo.plot(save=False)



# agglo = Agglomerative()
# agglo.load_data()
# for k in range(3, 31):
#     agglo.clustering(k, distance='custom')
#     print(k, agglo.get_validation_scores()[0][0])