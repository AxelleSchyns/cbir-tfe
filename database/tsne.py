import numpy as np
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import utils
import redis
import time
import seaborn as sns
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )
    args = parser.parse_args()

    index = faiss.read_index(args.db_name + '_labeled')
    r = redis.Redis(host='localhost', port='6379', db=0)
    # Retrieve the vectors from the index
    vectors = index.index.reconstruct_n(0, index.ntotal)

    labels = list(range(index.ntotal))
    names = []
    for l in labels:
        n = r.get(str(l) + 'labeled').decode('utf-8')
        names.append(utils.get_class(n))    
    
    classes = list(set(names))
    conversion = {x: i for i, x in enumerate(classes)}
    int_names = np.array([conversion[n] for n in names])
    # Perform t-SNE on the vectors
    tsne = TSNE(n_components=2, perplexity = 30, method = 'barnes_hut')
    #tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, metric='euclidean')
    
    t = time.time()
    embeddings = tsne.fit_transform(vectors)
    t_fit = time.time() - t
    print(t_fit)

    sns.scatterplot(embeddings[:,0], embeddings[:,1], hue=names)
    plt.show()
    # Visualize the embeddings
    plt.scatter(embeddings[:,0], embeddings[:,1], c=int_names,cmap='viridis')
    plt.colorbar()
    plt.show()


