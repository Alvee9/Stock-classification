from matplotlib import pyplot as plt
from utils.dataset import Dataset

def visualize_raw_data(ncompanies=30):
    dataset = Dataset()
    companies = dataset.read_company_names()
    closing_price = dataset.get_closing_prices(companies, ncompanies)
    for idx, r in enumerate(closing_price):
        plt.plot(r[1:], label = companies[int(r[0])])
    plt.title("Raw price time series")
    plt.legend()
    plt.show()

def visualize_normalized_data(ncompanies=30):
    dataset = Dataset()
    companies = dataset.read_company_names()
    closing_price = dataset.get_closing_prices(companies, ncompanies)
    normalized_price = dataset.normalize(closing_price)
    for idx, r in enumerate(normalized_price):
        plt.plot(r[1:], label = companies[int(r[0])])
    plt.title("Normalized price time series")
    plt.legend()
    plt.show()


from K_means.k_means import K_means
from BIRCH.birch import _Birch
from GMM.gmm import GMM
from Agglomerative.agglomerative import Agglomerative
from Spectral.spectral import Spectral


def plot_silhouette_graph():
    sil_scores = []
    k_means = K_means()
    k_means.load_data()
    for k in range(3, 31):
        k_means.clustering(k)
        sil_scores.append(k_means.get_validation_scores()[0][0])
    plt.plot([k for k in range(3, 31)], sil_scores, label='K-means')
    
    sil_scores = []
    brc = _Birch()
    brc.load_data()
    for k in range(3, 31):
        brc.clustering(k)
        sil_scores.append(brc.get_validation_scores()[0][0])
    plt.plot([k for k in range(3, 31)], sil_scores, label='BIRCH')

    sil_scores = []
    gmm = GMM()
    gmm.load_data()
    for k in range(3, 31):
        gmm.clustering(k)
        sil_scores.append(gmm.get_validation_scores()[0][0])
    plt.plot([k for k in range(3, 31)], sil_scores, label='GMM')

    sil_scores = []
    agglo = Agglomerative()
    agglo.load_data()
    for k in range(3, 31):
        agglo.clustering(k)
        sil_scores.append(agglo.get_validation_scores()[0][0])
    plt.plot([k for k in range(3, 31)], sil_scores, label='Agglomerative')

    sil_scores = []
    spectral = Spectral()
    spectral.load_data()
    for k in range(3, 31):
        spectral.clustering(k)
        sil_scores.append(spectral.get_validation_scores())
    plt.plot([k for k in range(3, 31)], sil_scores, label='Spectral')

    plt.title("Silhouette scores for each of the algorithms")
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.show()

plot_silhouette_graph()
