import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data = dataset.get_closing_prices(companies, 50)
data = dataset.normalize(data)
data = np.array(data)[:, 0:100]


def clustering(data, n_components=5):
    from sklearn.mixture import GaussianMixture
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score

    gm = GaussianMixture(n_components=n_components, random_state=0).fit(np.array(data)[:, 1:])

    labels = gm.predict(np.array(data)[:, 1:])

    sil_score = silhouette_score(np.array(data)[:, 1:], labels)
    print(sil_score)

    for k in range(0, n_components):
        clusterSize = 0
        for idx, r in enumerate(data):
            if labels[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()


for i in range(6, 7):
    print(i)
    clustering(data, i)
