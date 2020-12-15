import os
import logging
import random
import datetime
import argparse

import gensim
import pandas as pd

from gensim_w2v_edit import Word2Vec

export_dir = "./result"

print("start!")
root = "../data"
gene_set = "GDS5420"
targetfile = os.path.join(root, f"{gene_set}_co-exp.csv")
df_csv = pd.read_csv(targetfile)
gene_pairs = df_csv.values
gene_pairs = gene_pairs[:12000]
print(gene_pairs.shape)

current_time = datetime.datetime.now()
print(current_time)
print("shuffle start " + str(len(gene_pairs)))
random.shuffle(gene_pairs)
current_time = datetime.datetime.now()
print(current_time)
print("shuffle done " + str(len(gene_pairs)))

####training parameters########
dimension = 100  # dimension of the embedding
num_workers = 1  # number of worker threads
sg = 1  # sg =1, skip-gram, sg =0, CBOW
max_iter = 10  # number of iterations
window_size = 1  # The maximum distance between the gene and predicted gene within a gene list
txtOutput = True

for current_iter in range(1,max_iter+1):
    if current_iter == 1:
        print(f"gene2vec dimension {dimension} iteration {current_iter} start")
        # model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
        model = Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
        model.save(export_dir+f"gene2vec_dim_{dimension}_iter_{current_iter}")
        print(f"gene2vec dimension {dimension} iteration {current_iter} done")
        del model
    else:
        current_time = datetime.datetime.now()
        print(current_time)
        print("shuffle start " + str(len(gene_pairs)))
        random.shuffle(gene_pairs)
        current_time = datetime.datetime.now()
        print(current_time)
        print("shuffle done " + str(len(gene_pairs)))

        print(f"gene2vec dimension {dimension} iteration {current_iter} start")
        # model = gensim.models.Word2Vec.load(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter-1))
        model = Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
        model.train(gene_pairs,total_examples=model.corpus_count,epochs=model.iter)
        model.save(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter))
        print(f"gene2vec dimension {dimension} iteration {current_iter} done")
        del model
