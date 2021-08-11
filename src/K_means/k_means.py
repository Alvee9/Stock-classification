import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data_closing = dataset.get_closing_prices(companies, 30)
data_change = dataset.calc_price_changes(companies, 30)
data_closing = dataset.normalize(data_closing)
data_change = np.array(data_change)[:, 0:300]
data_closing = np.array(data_closing)[:, 0:300]

# data = dataset.average_ndays(data, 7)

def clustering(data_closing, data_change, number_of_clusters):
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=50).fit(np.array(data_closing)[:, 1:])

    # Get the cluster centroids
    # print(kmeans.cluster_centers_)
        
    # Get the cluster labels
    # print(kmeans.labels_)

    sil_score = silhouette_score(np.array(data_closing)[:, 1:], kmeans.labels_)
    dbi = davies_bouldin_score(np.array(data_closing)[:, 1:], kmeans.labels_)
    chs = calinski_harabasz_score(np.array(data_closing)[:, 1:], kmeans.labels_)
    print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)

    for k in range(0, number_of_clusters):
        plt.clf()
        clusterSize = 0
        for idx, r in enumerate(data_closing):
            if kmeans.labels_[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()
        # plt.savefig('../Plots/K_means/{}.png'.format(k), dpi=600)



for i in range(8, 9):
    print('k =', i)
    clustering(data_closing, data_change, i)


