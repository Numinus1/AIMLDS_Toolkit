import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from scipy.cluster.hierarchy import linkage, dendrogram

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import chardet

from gensim.models import Word2Vec

import multiprocessing

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

seed = 420

def lemmatizeAndScrubStopWords(unified_text):
    
    lem = WordNetLemmatizer()

    unified_text = re.sub('[^A-Za-z0-9]+', ' ', unified_text)
    
    words = word_tokenize(unified_text)
    stop_words = set(stopwords.words('english'))

    lem_words = [lem.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(lem_words)

def preprocessTextDF(df):

    d = df.copy()
    d["Preprocessed"] = (d["Topic"].str.lower() + ' ' + d["YourAnalysis"].str.lower()).apply(lemmatizeAndScrubStopWords)

    return d

def vectorizeDocumentDF(d, matrices):

    vectorizers = {"CV": CountVectorizer(),
              "TfidV": TfidfVectorizer()}
    documents = d["Preprocessed"].tolist()
    
    for title, vectorizer in vectorizers.items():
       
        vectorizer.fit(documents)
    
        vectors = []
        for i in range(0, d.shape[0]):
            m = vectorizer.transform([d["Preprocessed"][i]])
            vectors.append(m.toarray()[0])
    
        d[title] = pd.Series(vectors)
        matrices[title] = vectorizer.transform(d["Preprocessed"]).toarray()

    return d, matrices

def word2VecDocumentDF(d, matrices):

    vsize = 100
    
    tokensByRow = [row.split() for row in d["Preprocessed"]]
    w2v = Word2Vec(tokensByRow, vector_size = vsize, window = 5, min_count = 1, workers = multiprocessing.cpu_count())

    v_docs = []

    for tokenRow in tokensByRow:
        
        v_w = []
        
        for token in tokenRow:
            if token in w2v.wv:
                v_w.append(w2v.wv[token])

        v_docs.append(np.mean(v_w, axis = 0) if len(v_w) > 0 else np.zeroes(vsize))

    print("Vocabulary Size: ", len(w2v.wv))

    matrices["W2V"] = np.array(v_docs)

    return d, matrices

def computeVectorizerDistances(d, matrices):

    fig, axs = plt.subplots(len(matrices.keys()) * 4, 2, figsize=(15, 4 * len(matrices.keys()) * 6))

    row = 0
    for title, matrix in matrices.items():
        
        cos_dist = cosine_distances(matrix)
        euc_dist = euclidean_distances(matrix)
        
        pca = PCA(n_components = 2)
        pca100 = PCA(n_components = 100)
        matrix_pca = pca.fit_transform(matrix)
        matrix_pca100 = pca100.fit_transform(matrix)
        tsne = TSNE(n_components = 2, random_state = seed)
        matrix_tsne = tsne.fit_transform(matrix_pca100)

        sil = []
        db = []

        for i in range(3, 16):
            
            clusterer = AgglomerativeClustering(n_clusters = i, linkage = "complete")
            labels = clusterer.fit_predict(matrix_tsne)

            sil.append(silhouette_score(matrix_tsne, labels))
            db.append(davies_bouldin_score(matrix_tsne, labels))
    
        im = axs[row, 0].imshow(cos_dist, cmap = 'plasma')
        fig.colorbar(mappable = im, ax = axs[row, 0], orientation = 'vertical', label = "Distance")
        axs[row, 0].set_title(f"{title}: Cosine Distances Between Documents")

        im = axs[row, 1].imshow(euc_dist, cmap = 'inferno')
        fig.colorbar(mappable = im, ax = axs[row, 1], orientation = 'vertical', label = "Distance")
        axs[row, 1].set_title(f"{title}: Euclidean Distances Between Documents")

        axs[row + 1, 0].scatter(matrix_pca[:, 0] ,matrix_pca[:, 1], s = 5, alpha = .4)
        axs[row + 1, 0].set_xlabel("PC1")
        axs[row + 1, 0].set_ylabel("PC2")
        axs[row + 1, 0].set_title(f"{title}: PCA Scatterplot")

        axs[row + 1, 1].scatter(matrix_tsne[:, 0] ,matrix_tsne[:, 1], s = 5, alpha = .4)
        axs[row + 1, 1].set_title(f"{title}: t-SNE Scatterplot")

        axs[row + 2, 0].plot(range(3, 16), sil, marker = "*")
        axs[row + 2, 0].set_xlabel("Clusters")
        axs[row + 2, 0].set_ylabel("Silhouette Scores")
        axs[row + 2, 0].set_title(f"{title}: Silhouette Scores")

        axs[row + 2, 1].plot(range(3, 16), db, marker = "*")
        axs[row + 2, 1].set_xlabel("Clusters")
        axs[row + 2, 1].set_ylabel("Davis Bouldin Scores")
        axs[row + 2, 1].set_title(f"{title}: Davis Bouldin Scores")

        dendrogram(Z = linkage(matrix_tsne, method = "complete"), truncate_mode = "level", p = 5, ax = axs[row + 3, 0])
        axs[row + 3, 0].set_xlabel("Document")
        axs[row + 3, 0].set_ylabel("Distance")
        axs[row + 3, 0].set_title(f"{title}: Dendrogram")

        # Silhouette maxes ~ 15, DB mins ~ 15; using k = 15; following created based on this

        cluster_optimal = 15
        
        clusterer_ = AgglomerativeClustering(n_clusters = cluster_optimal, linkage = "complete")
        clusters = clusterer_.fit_predict(matrix_tsne)

        cluster_counts = []
        for k in range(0, cluster_optimal):
            cluster_counts.append(clusters.tolist().count(k))

        axs[row + 3, 1].bar([f"C{n}" for n in range(0, cluster_optimal)], cluster_counts)
        axs[row + 3, 1].grid(True)
        axs[row + 3, 1].set_xlabel("Cluster")
        axs[row + 3, 1].set_ylabel("Frequency")
        axs[row + 3, 1].set_title(f"{title}: Cluster Bar Chart")

        d[f"{title} Clusters"] = clusters
        
        row += 4