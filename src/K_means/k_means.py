import numpy as np
from utils.dataset import Dataset


class K_means:
    def __init__(self) -> None:
        self.dataset = Dataset()

        self.companies = self.dataset.read_company_names()
        self.data_closing = self.dataset.get_closing_prices(self.companies, 200)
        self.data_change = self.dataset.calc_price_changes(self.companies, 200)
        self.data_closing = self.dataset.normalize(self.data_closing)
        self.data_change = np.array(self.data_change)[:, 0:300]
        self.data_closing = np.array(self.data_closing)[:, 0:300]

        self.number_of_clusters = 5
        self.cluster_labels = []
        self.cluster_sizes = []

    
    def clustering(self, number_of_clusters):
        from sklearn.cluster import KMeans

        self.number_of_clusters = number_of_clusters

        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0, n_init=50).fit(np.array(self.data_closing)[:, 1:])

        self.cluster_labels = list(kmeans.labels_)
        
        for k in range(0, self.number_of_clusters):
            clusterSize = 0
            for idx, r in enumerate(self.data_closing):
                if self.cluster_labels[idx] == k:
                    clusterSize += 1
            self.cluster_sizes.append(clusterSize)
        

    def get_validation_scores(self):
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        sil_score = silhouette_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        dbi = davies_bouldin_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        chs = calinski_harabasz_score(np.array(self.data_closing)[:, 1:], self.cluster_labels)
        # print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)

        return [sil_score, dbi, chs]


    def plot(self):
        from matplotlib import pyplot as plt

        for k in range(0, self.number_of_clusters):
            plt.clf()
            clusterSize = 0
            for idx, r in enumerate(self.data_closing):
                if self.cluster_labels[idx] == k:
                    plt.plot(r[1:], label = self.companies[int(r[0])])
                    clusterSize += 1
            plt.title('Cluster number {} size = {}'.format(k, clusterSize))
            plt.legend()
            # plt.show()
            plt.savefig('../Plots/K_means/{}.png'.format(k), dpi=600)


k_means = K_means()

for k in range(37, 38):
    print('k =', k)
    k_means.clustering(k)
    print(k_means.get_validation_scores())
    k_means.plot()