import numpy as np

from K_means.k_means import K_means
from BIRCH.birch import _Birch
from GMM.gmm import GMM
from Agglomerative.agglomerative import Agglomerative
from Spectral.spectral import Spectral


def birch_kmeans():
    _birch = _Birch()
    _birch.load_data()
    _birch.clustering(20)
    _birch.plot('../Plots/BIRCH_K_means/{}.png', save=True)

    for i in range(0, _birch.number_of_clusters):
        if _birch.cluster_sizes[i] > 10:
            data = []
            for j in _birch.cluster_lists[i]:
                data.append(_birch.data_closing[j])
            div_k_means = K_means()
            div_k_means.data_closing = data
            div_k_means.clustering(int(np.ceil(_birch.cluster_sizes[i]/10.0)))
            div_k_means.plot('../Plots/BIRCH_K_means/' + str(i) + '_{}.png', save=True)

def gmm_agglo():
    gmm = GMM()
    gmm.load_data()
    gmm.clustering(20)
    gmm.plot('../Plots/GMM_Agglo/{}.png', save=True)
    for i in range(0, gmm.number_of_clusters):
        if gmm.cluster_sizes[i] > 10:
            print("original cluster", i)
            data = []
            for j in gmm.cluster_lists[i]:
                data.append(gmm.data_closing[j])
            div_agglo = Agglomerative()
            div_agglo.data_closing = data
            # div_agglo.distance_matrix = get_dist_matrix(data)
            div_agglo.clustering(int(np.ceil(gmm.cluster_sizes[i]/10.0)))
            div_agglo.plot('../Plots/GMM_Agglo/' + str(i) + '_{}.png', save=True)


def spectral_spectral():
    spectral = Spectral()
    spectral.load_data()
    spectral.clustering(20, distance='custom')
    spectral.plot('../Plots/Spectral_Spectral/{}.png', save=True)
    for i in range(0, spectral.number_of_clusters):
        if spectral.cluster_sizes[i] > 10:
            print("original cluster", i)
            data = []
            for j in spectral.cluster_lists[i]:
                data.append(spectral.data_closing[j])
            div_spectral = Spectral()
            div_spectral.data_closing = data
            # div_spectral.distance_matrix = get_edge_matrix(data)
            div_spectral.clustering(int(np.ceil(spectral.cluster_sizes[i]/10.0)))
            div_spectral.plot('../Plots/Spectral_Spectral/' + str(i) + '_{}.png', save=True)


spectral_spectral()