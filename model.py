import faiss
from train_faiis import load_sentence_vectors
from utils import build_model, sent_to_vec, load_whiten, save_whiten,transform_and_normalize
import pickle
import torch
import numpy as np
import time

class PredictModel:
    def __init__(self,model_path,w_b_path,pkl_path,index_path):
        self.device = torch.device("cuda:1")
        torch.cuda.set_device(self.device)
        self.tokenizer, self.model = build_model(model_path, self.device)
        self.max_length = 32
        self.pooling = 'first_last_avg'

        self.w ,self.b = load_whiten(w_b_path)

        _, sentence = load_sentence_vectors(pkl_path)
        self.id_sentence = {k:v for k,v in enumerate(sentence)}
        self.faiss_search = self.load_faiss_index(index_path)


    def vector_query(self,text):
        vec = sent_to_vec(text, self.tokenizer, self.model, self.pooling, self.max_length, self.device)
        vec = transform_and_normalize(vec,self.w,self.b)
        return vec
    def load_faiss_index(self,index_path,gpu=1):
        self.index = faiss.read_index(index_path)
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
        return self.index

    def search(self,text,top=10,nprobe = 1000):
        self.index.nprobe = nprobe
        vec = self.vector_query(text)
        vec = vec.astype("float32")
        start_time = time.time()
        D,I = self.index.search(vec,top)
        end_time = time.time()
        print('search time costs :',end_time-start_time)
        res = []
        for index,distances in zip(I,D):
            for id,dis in zip(index,distances):
                sentence = self.id_sentence[id]
                res.append((sentence,dis))
        print(res)





if __name__ == "__main__":
    model_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-base-nli-mean-tokens'
    w_b_path = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/bert-base-nli-mean-tokens-first_last_avg-whiten(NLI).pkl'
    pkl_path = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/vctors_sentence.pkl'
    index_path = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/faiss.index'
    predictor = PredictModel(model_path,w_b_path,pkl_path,index_path)
    text = "我想要美女的微信"
    predictor.search(text)














