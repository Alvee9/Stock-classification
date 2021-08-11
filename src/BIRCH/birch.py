import numpy as np
from scipy.sparse import data
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data_closing = dataset.get_closing_prices(companies, 200)
data_change = dataset.calc_price_changes(companies, 200)
data_closing = dataset.normalize(data_closing)
data_change = np.array(data_change)[:, 0:300]
data_closing = np.array(data_closing)[:, 0:300]


def clustering(data_closing, data_change, number_of_clusters=5):
    from sklearn.cluster import Birch
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    brc = Birch(n_clusters=number_of_clusters)
    brc.fit(np.array(data_closing)[:, 1:])

    labels = brc.predict(np.array(data_closing)[:, 1:])

    sil_score = silhouette_score(np.array(data_closing)[:, 1:], labels)
    dbi = davies_bouldin_score(np.array(data_closing)[:, 1:], labels)
    chs = calinski_harabasz_score(np.array(data_closing)[:, 1:], labels)
    print('silhouette=', sil_score, 'dbi=', dbi, 'chs=', chs)

    # for k in range(0, number_of_clusters):
    #     plt.clf()
    #     clusterSize = 0
    #     for idx, r in enumerate(data_closing):
    #         if labels[idx] == k:
    #             plt.plot(r[1:], label = companies[int(r[0])])
    #             clusterSize += 1
    #     plt.title('Cluster number {} size = {}'.format(k, clusterSize))
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('../Plots/BIRCH/{}.png'.format(k), dpi=600)


for i in range(3, 40):
    print(i)
    clustering(data_closing, data_change, i)
