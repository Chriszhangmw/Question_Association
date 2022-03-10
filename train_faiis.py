import os
import faiss
import pickle
import numpy as np
import collections
from utils import build_model, sent_to_vec, http_get, load_whiten
from utils import compute_kernel_bias, transform_and_normalize, normalize
from utils import get_size
from tqdm import tqdm
import time




# FAISS config
top_k_hits = 10
n_clusters = 1024
nprobe = 5



class Faiss:
    def __init__(self,dim=256,nlist=1000,gpu=1):
        self.dim = dim
        self.nlist = nlist
        self.quentizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(self.quentizer,self.dim,self.nlist,faiss.METRIC_INNER_PRODUCT)
        try:
            if gpu >= 0:
                if gpu == 0:
                    source = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(source,0,self.index)
                else:
                    gpu_index = faiss.index_cpu_to_gpu(self.index)
                self.index = gpu_index
        except:
            print("error to load gpu")
        self.xb = None

    def train(self,vectors,faiss_index_path):
        if not vectors.dtype == "float32":
            vectors = vectors.astype("float32")
        if self.xb is None:
            self.xb = vectors.copy()
        self.index.train(self.xb)
        self.index.add(self.xb)
        faiss.write_index(self.index,faiss_index_path)
        print("faiss training done ")


def load_sentence_vectors(embedding_cache_path):
    with open(embedding_cache_path, 'rb') as fin:
        sentence_vectors = pickle.load(fin)
    # Format sentence vectors
    vecs = np.vstack([v for _, v in sentence_vectors.items()])
    sentence = np.array([i for i, _ in sentence_vectors.items()])
    return vecs, sentence


def main():
    pkl_path = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/vctors_sentence.pkl'
    index_path = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/faiss.index'
    # Load vectors
    vecs, sentence = load_sentence_vectors(pkl_path)
    vecs = vecs.astype('float32')
    print("Docs num:", len(vecs))
    vecs_memory_size = get_size(vecs)
    print("vecs Memory Size: {} GB".format(vecs_memory_size / 1073741824.0))
    # Create Index
    f = Faiss()
    f.train(vecs,index_path)


if __name__ == "__main__":
    main()