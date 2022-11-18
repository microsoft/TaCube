# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for i, text in enumerate(paragraphs):
        corpus.append(text)
        sorted_order.append(i)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x:x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx], sorted_similarities
