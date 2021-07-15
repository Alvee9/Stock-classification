import pandas as pd
import numpy as np

def read_company_names():
    companies = []
    with open('Dataset/s&p500_symbols.txt') as f:
        lines = f.read().split()
        for l in lines:
            companies.append(l)
    return companies

companies = read_company_names()


def get_closing_prices(companies, number_of_companies):
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
        if len(data) == number_of_companies:
            break
    return data


def calc_price_changes(companies, number_of_companies):
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

        if len(data) == number_of_companies:
            break
    return data


def average_ndays(data, n):
    new_data = []
    for trendline in data:
        sum = 0.0
        new_trendline = [trendline[0]]
        for i in range(1, len(trendline)):
            sum += trendline[i]
            divisor = i
            if i > n:
                divisor = n
                sum -= trendline[i - n]
            new_trendline.append(sum / divisor)
        new_data.append(new_trendline)
    return new_data


# data = get_closing_prices(companies, 50)
data = calc_price_changes(companies, 50)
data = np.array(data)[:, 0:60]
print(len(data))
data = average_ndays(data, 7)

def clustering(number_of_clusters):
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=50).fit(np.array(data)[:, 1:])

    # Get the cluster centroids
    print(kmeans.cluster_centers_)
        
    # Get the cluster labels
    print(kmeans.labels_)
    print(len(kmeans.labels_))

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