import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data = dataset.get_closing_prices(companies, 100)
data = dataset.normalize(data)
data = np.array(data)[:, 0:300]


print(len(data))
# data = average_ndays(data, 7)

def clustering(eps, min_samples):
    from matplotlib import pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(data)[:, 1:])
        
    # Get the cluster labels
    # print(dbscan.labels_)
    number_of_labels = len(set(dbscan.labels_))
    if number_of_labels == len(data) or number_of_labels == 1:
        return
    
    cluster_count = len(dbscan.labels_)
    print('number of labels', number_of_labels)
    
    sil_score = silhouette_score(data, dbscan.labels_)
    print('silhouette score', sil_score)

    for k in range(-1, number_of_labels):
        clusterSize = 0
        for idx, r in enumerate(data):
            if dbscan.labels_[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()


for e in range(115, 116):
    for m in range(3, 4):
        print('\neps = ', e, 'minPts = ', m)
        clustering(e, m)