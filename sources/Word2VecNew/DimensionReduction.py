from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

RS = None # Random state

# Алгоритм снижения размерности до 2
# Латентно-семантический анализ (снижения размерности до 50, либо для визуализации)
def lsa_algorithm(X, n):
    lsa = TruncatedSVD(n_components=n, random_state=RS)
    return lsa.fit_transform(X)

def tsne_algorithm(X, n):
    tsne = TSNE(n_components=n, perplexity = 50, random_state=RS)
    return tsne.fit_transform(X)

# Метод главных компонент
def pca_algorithm(X, n):
    # Снижаем размерность до n
    pca = PCA(n_components=n, random_state=RS)
    return pca.fit_transform(X)

def tsvd_tsne_algorithms(X):
    lsa = lsa_algorithm(X, 50)
    tsne = tsne_algorithm(lsa, 2)
    return tsne

def pca_tsne_algorithm(X):
    data = X
    if(X.shape[0] > 50):
        data = pca_algorithm(X, 50)
    tsne = tsne_algorithm(data, 2)
    return tsne
