import numpy as np
from utils.dataset import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from matplotlib import pyplot as plt

class GMM:
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
        gm = GaussianMixture(n_components=number_of_clusters, random_state=0).fit(np.array(self.data_closing)[:, 1:])
        self.cluster_labels = list(gm.predict(np.array(self.data_closing)[:, 1:]))
        
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


    def plot(self, path='../Plots/GMM/{}.png', save=False):
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
    gmm = GMM()
    gmm.load_data()
    gmm.clustering(20)

    gmm.plot('../Plots/GMM/{}.png', save=True)


    for i in range(0, gmm.number_of_clusters):
        if gmm.cluster_sizes[i] > 7:
            print("original cluster", i)
            data = []
            for j in gmm.cluster_lists[i]:
                data.append(gmm.data_closing[j])
            div_gmm = GMM()
            div_gmm.data_closing = data
            div_gmm.clustering(int(np.ceil(gmm.cluster_sizes[i]/7.0)))
            div_gmm.plot(save=False)

    for k in range(3, 31):
        gmm = GMM()
        gmm.load_data()
        gmm.clustering(k)
        print(k, gmm.get_validation_scores()[0][0])