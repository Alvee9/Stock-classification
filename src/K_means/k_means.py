import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data = dataset.get_closing_prices(companies, 50)
data = dataset.normalize(data)
data = np.array(data)[:, 0:100]


data = np.array(data)[:, 0:100]
data = dataset.average_ndays(data, 7)

def clustering(number_of_clusters):
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=50).fit(np.array(data)[:, 1:])

    # Get the cluster centroids
    # print(kmeans.cluster_centers_)
        
    # Get the cluster labels
    print(kmeans.labels_)

    for k in range(0, number_of_clusters):
        clusterSize = 0
        for idx, r in enumerate(data):
            if kmeans.labels_[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()


clustering(10)