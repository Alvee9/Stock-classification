import numpy as np
from utils.dataset import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from matplotlib import pyplot as plt

class K_means:
    def __init__(self) -> None:
        self.dataset = Dataset()
        self.companies = self.dataset.read_company_names()
        self.number_of_clusters = 2
        self.data_closing = []
        self.data_change = []
        self.cluster_labels = []
        self.cluster_sizes = []
        self.clusters_list = []


    def load_data(self):
        self.companies = self.dataset.read_company_names()
        self.data_closing = self.dataset.get_closing_prices(self.companies, 200)
        self.data_change = self.dataset.calc_price_changes(self.companies, 200)
        self.data_closing = self.dataset.normalize(self.data_closing)
        self.data_change = np.array(self.data_change)[:, 0:300]
        self.data_closing = np.array(self.data_closing)[:, 0:300]
    
    def clustering(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0, n_init=50).fit(np.array(self.data_closing)[:, 1:])
        self.cluster_labels = list(kmeans.labels_)
        
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


        return ([sil_score, dbi, chs], silhouette_samples(np.array(self.data_closing)[:, 1:], self.cluster_labels))


    def plot(self, path='../Plots/K_means/{}.png'):
        for k in range(0, self.number_of_clusters):
            plt.clf()
            clusterSize = 0
            for idx, r in enumerate(self.data_closing):
                if self.cluster_labels[idx] == k:
                    plt.plot(r[1:], label = self.companies[int(r[0])])
                    clusterSize += 1
            plt.title('Cluster number {} size = {}'.format(k, clusterSize))
            plt.legend()
            plt.show()
            # plt.savefig(path.format(k), dpi=600)


k_means = K_means()
k_means.load_data()
k_means.clustering(30)

# k_means.plot('../Plots/K_means/{}.png')

# validation = k_means.get_validation_scores()

# sil_avg = []

# for k in range(0, k_means.number_of_clusters):
#     sil_sum = 0
#     for i in k_means.cluster_lists[k]:
#         sil_sum += validation[1][i]
#     sil_avg.append(sil_sum / k_means.cluster_sizes[k])


for i in range(0, k_means.number_of_clusters):
    if k_means.cluster_sizes[i] > 7:
        print("original cluster", i)
        data = []
        for j in k_means.cluster_lists[i]:
            data.append(k_means.data_closing[j])
        div_k_means = K_means()
        div_k_means.data_closing = data
        div_k_means.clustering(int(np.ceil(k_means.cluster_sizes[i]/7.0)))
        div_k_means.plot()

