import pandas as pd



def read_company_names():
    companies = []
    with open('Dataset/s&p500_symbols.txt') as f:
        lines = f.read().split()
        for l in lines:
            companies.append(l)
    return companies

companies = read_company_names()

def get_closing_prices(companies):
    data = []
    for idx, company in enumerate(companies):
        # df = pd.read_csv('Dataset/Comp_time_series/{}.csv'.format(company))
        try:
            df = pd.read_csv('Dataset/Comp_time_series/{}.csv'.format(company))
            if len(df['Close']) != 753:
                continue
            data.append([idx] + list(df['Close']))
        except FileNotFoundError:
            print(company, 'Not found')
        if len(data) == 100:
            break
    return data

def calc_price_changes(companies):
    data = []
    for idx, company in enumerate(companies):
        try:
            df = pd.read_csv('Dataset/Comp_time_series/{}.csv'.format(company))
            if len(df['Close']) != 753:
                continue
        except FileNotFoundError:
            print(company, 'Not found')
            continue
        price_changes = [idx] + [0] * len(df['Close'])
        for i in range(1, len(df['Close'])):
            price_changes[i] = (df['Close'][i] - df['Close'][i - 1]) / df['Close'][i - 1] * 100
        data.append(price_changes)

        if len(data) == 100:
            break
    return data

data = calc_price_changes(companies)

print(len(data))

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0, n_init=50).fit(np.array(data)[:, 1:])

# Get the cluster centroids
print(kmeans.cluster_centers_)
    
# Get the cluster labels
print(kmeans.labels_)
print(len(kmeans.labels_))

# Plotting the cluster centers and the data points on a 2D plane
# plt.scatter(data[:][0], data[:][1])
    
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    
# plt.title('Data points and cluster centroids')
# plt.show().

for k in range(1, 10):
    clusterSize = 0
    for idx, r in enumerate(data):
        if kmeans.labels_[idx] == k:
            plt.plot(r[1:], label = companies[r[0]])
            clusterSize += 1
    plt.title('Cluster number {} size = {}'.format(k, clusterSize))
    plt.legend()
    plt.show()

