import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data = dataset.get_closing_prices(companies, 100)
data = dataset.normalize(data)
data = np.array(data)[:, 0:200]


def clustering(data, number_of_clusters=5):
    from sklearn.cluster import Birch
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score

    brc = Birch(n_clusters=number_of_clusters)
    brc.fit(data)

    labels = brc.predict(data)

    sil_score = silhouette_score(data, labels)
    print(sil_score)

    for k in range(0, number_of_clusters):
        clusterSize = 0
        for idx, r in enumerate(data):
            if labels[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()


for i in range(19, 20):
    print(i)
    clustering(data, i)
