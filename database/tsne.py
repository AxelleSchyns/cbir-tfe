import numpy as np
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )
    args = parser.parse_args()

    index = faiss.read_index(args.db_name + '_labeled')

    # Retrieve the vectors from the index
    """vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        vectors[i] = index.reconstruct(i)"""
    vectors = index.reconstruct_n(list(range(index.ntotal)))
    if isinstance(index, faiss.IndexIDMap):
        _, labels = index.reconstruct_n(0,index.ntotal)

    # Perform t-SNE on the vectors
    tsne = TSNE(n_components=2, perplexity = 30, solver = 'barnes_hut', backend = 'cublas')
    #tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, metric='euclidean')
    embeddings = tsne.fit_transform(vectors)

    # Visualize the embeddings
    plt.scatter(embeddings[:,0], embeddings[:,1], c=labels,cmap='viridis')
    plt.colorbar()
    plt.show()


