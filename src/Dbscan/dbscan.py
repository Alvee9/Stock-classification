import numpy as np
from utils.dataset import Dataset

dataset = Dataset()

companies = dataset.read_company_names()

data = dataset.get_closing_prices(companies, 50)
data = dataset.normalize(data)
data = np.array(data)[:, 0:100]


print(len(data))
# data = average_ndays(data, 7)

def clustering():
    from matplotlib import pyplot as plt
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=20, min_samples=1).fit(np.array(data)[:, 1:])
        
    # Get the cluster labels
    print(dbscan.labels_)
    cluster_count = len(dbscan.labels_)
    

    for k in range(0, 20):
        clusterSize = 0
        for idx, r in enumerate(data):
            if dbscan.labels_[idx] == k:
                plt.plot(r[1:], label = companies[int(r[0])])
                clusterSize += 1
        plt.title('Cluster number {} size = {}'.format(k, clusterSize))
        plt.legend()
        plt.show()


clustering()