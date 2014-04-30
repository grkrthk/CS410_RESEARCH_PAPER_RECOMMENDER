# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import os
import shutil

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   "to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set
categories = [
    'file.system.storage.cache.IO.io.memory.ssd.flash.kernel.deduplication.algorithms', # fast
    'file.system.storage.cache.IO.io.memory.ssd.flash.kernel.deduplication.algorithms', # hotstorage
    'distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system.network', #lisa
    'distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system.network', #nsdi
    'distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system.network', #osdi
    'distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system.network', #sacan
    'network.sdn.SDN.protocols.IP.packet.router.topology.flow.openflow.flow', #sigcomm
    'distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system', #sosp
    'file.system.storage.cache.IO.io.memory.ssd.flash.kernel.deduplication.distributed.systems.hadoop.map.reduce.peerp.p2p.cloud.stream.scheduling.operating.kernel.system.network.network.sdn.SDN.protocols.IP.packet.router.topology.flow.openflow' #usenix
]


#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 fast
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 hotstorage
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 lisa
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 nsdi
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 osdi
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 sacan
#drwxrwxr-x 2 grk grk 36864 Apr 30 15:18 sigcomm
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 sosp
#drwxrwxr-x 2 grk grk  4096 Apr 30 15:18 usenix

# Uncomment the following to do the analysis on all the categories
#categories = None

print(categories)

#dataset = fetch_20newsgroups(subset='all', categories=categories,
#                             shuffle=True, random_state=42)

dataset = load_files("/home/grk/cs410_project/parsed_text/")

#dataset_files = load_files("./");

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = Pipeline((
            ('hasher', hasher),
            ('tf_idf', TfidfTransformer())
        ))
    else:
        vectorizer = HashingVectorizer(n_features=10000,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.9, max_features=opts.n_features,
                                 stop_words='english', use_idf=opts.use_idf)

#print(vectorizer.get_feature_names())

print("%d",vectorizer);
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    lsa = TruncatedSVD(opts.n_components)
    X = lsa.fit_transform(X)
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    X = Normalizer(copy=False).fit_transform(X)

    print("done in %fs" % (time() - t0))
    print()

###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=10, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
     

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, sample_size=1000))

print()

clusters = dict()

k = 0;
for i in km.labels_ :

  if i in clusters.keys():
        clusters[i].append(dataset.filenames[k])
  else:
        clusters[i] = [dataset.filenames[k]]
  #print("-------------\n")
  #print("%d" % i)
  #print("%s" % dataset.filenames[k])
  #print("-------------\n")
  k +=  1

#print("Total no of document is %d" % k)
i = 0
for key, value in clusters.iteritems() :
           print("%d , %s" %(key, value))
           print("\n")
           make_dir_path = "/home/grk/cs410_project/clustered_results" + "/cluster" + str(i)
           dir_path = make_dir_path + "/cluster_folder"
                                     
           try:
                          os.makedirs(make_dir_path)                                         
                          os.makedirs(dir_path)
           except OSError:
                          pass

           for path in value:                          
                          shutil.copy(path, dir_path)
                          with open(path, "r") as myfile:
                                  lines = myfile.readlines()				  
                                  last_line = lines[-1]
                                  with open(dir_path + "/" + "unique_ids.txt", "a") as myfile:
                                                   myfile.write(last_line)
                     
           i = i+1

sum1 = 0
count_clusters = dict()
for key, value in clusters.iteritems() :
           print("key: %d, total: %d" %(key, len(value)))
           sum1 = sum1 + len(value)
           count_clusters[key] = len(value)
  
print("total docs: %d" %(sum1))

#print(vectorizer.get_feature_names())

#for clust in clusters :
  #print("\n************************\n")
#  print("%s" % clusters[clust])

#for cluster_id in range(0, km.n_clusters):
#    cluster_doc_filenames = dataset.filenames[cluster_id]
#    print("%s" % cluster_doc_filenames)

path = "/home/grk/cs410_project/clustered_results/cluster"

for i in range(0, km.n_clusters):
             new_path = path + str(i)
             command = "python" + " topic_extraction.py" + " " + new_path + "/ " + str(count_clusters[i])
             print(command)
             os.system(command)


resultant_path = "/home/grk/cs410_project/resultant.txt"
with open(resultant_path, "a") as resfile:
  
       for i in range(0, km.n_clusters):
       		      unique_file_id_path = path + str(i) + "/" + "cluster_folder/" + "unique_ids.txt"
	              topic_keywords_path = path + str(i) + "/" + "cluster_folder/" + "topic_keywords.txt"
                      topicfile = open(topic_keywords_path, "r") 
	              keywords_buf = topicfile.readlines()         
              
                      uniqueidfile = open(unique_file_id_path, "r")
        	      uniqueidlines = uniqueidfile.readlines()
                     
                      for line in uniqueidfile:
                                     line = line + ":" + keywords_buf
                                     resfile.write(line + "\n")

                                     
                                 
             
