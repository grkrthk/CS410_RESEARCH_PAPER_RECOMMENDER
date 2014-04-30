# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function

from time import time
from sklearn.feature_extraction import text
from sklearn import decomposition
from sklearn import datasets
from sklearn.datasets import load_files

import sys
cmdargs = sys.argv
n_samples = 10000
n_features = 10000
n_topics = int(cmdargs[2])
n_top_words = 4

# frequency with TF-IDF weighting (without top 5% stop words)

t0 = time()
print("Loading dataset and extracting TF-IDF features...")
#dataset = datasets.fetch_20newsgroups(shuffle=True, random_state=1)

parse_path = str(cmdargs[1])
print(parse_path)
dataset = load_files(parse_path)

vectorizer = text.CountVectorizer(max_df=0.55, max_features=n_features)
counts = vectorizer.fit_transform(dataset.data[:n_samples])
tfidf = text.TfidfTransformer().fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model on with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = decomposition.NMF(n_components=n_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()
feature_list = [ ]
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))

    for i in topic.argsort()[:-n_top_words - 1:-1] :
                     feature_list.append(feature_names[i]) 
    print()


final_path = parse_path + "cluster_folder" + "/topic_keywords.txt"

with open(final_path, "a") as myfile:
     str1 = ','.join(str(e) for e in feature_list)
     myfile.write(str1)

    
print(feature_list)
