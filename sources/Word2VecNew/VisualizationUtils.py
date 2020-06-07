from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Отображение tsne 
def tsne_convert_algorithm(self):
        RS = 2500 # Random state
        tsne = TSNE(n_components=2, perplexity = 40, random_state=RS)
        result = tsne.fit_transform(X)

def pda_convert_algorithm():
    pca = PCA(n_components=2)
    