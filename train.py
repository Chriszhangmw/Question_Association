import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import scipy.stats
import pickle
from utils import build_model, sents_to_vecs, compute_kernel_bias, save_whiten,transform_and_normalize
import pickle



NLI_PATH = '/home/zmw/projects/question_matching/sourceData/data_engineering/train_eda_t_ratio_bak.csv'


MODEL_NAME_LIST = [
    # '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-base-uncased',
    # '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-large-uncased',
    '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-base-nli-mean-tokens'
    # '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-large-nli-mean-tokens'
]

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

MAX_LENGTH = 32
OUTPUT_DIR = '/home/zmw/big_space/zhangmeiwei_space/nlp_out/bert_whiten/'


def load_dataset(path):
    """
    loading AllNLI dataset.
    """
    senta_batch, sentb_batch = [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            # if i < 500:
            items = line.strip().split('\t')
            senta, sentb = items[0], items[1]
            senta_batch.append(senta)
            sentb_batch.append(sentb)
    return senta_batch, sentb_batch


def main():

    a_sents_train, b_sents_train  = load_dataset(NLI_PATH)
    print("Loading {} training samples from {}".format(len(a_sents_train), NLI_PATH))

    device = torch.device("cuda:1")
    torch.cuda.set_device(device)

    # for MODEL_NAME in MODEL_NAME_LIST:
    MODEL_NAME = '/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/bert-base-nli-mean-tokens'
    tokenizer, model = build_model(MODEL_NAME,device)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

    print("Transfer sentences to BERT vectors.")
    a_vecs_train = sents_to_vecs(a_sents_train, tokenizer, model, POOLING, MAX_LENGTH,device)
    b_vecs_train = sents_to_vecs(b_sents_train, tokenizer, model, POOLING, MAX_LENGTH,device)

    # 存储检索的向量本身
    sentence = np.append(a_sents_train,b_sents_train)
    vectors = np.concatenate((a_vecs_train,b_vecs_train))

    print("Compute kernel and bias.")
    kernel, bias = compute_kernel_bias(vectors)
    print("finish kernel and bias.")

    model_name = MODEL_NAME.split('/')[-1]
    output_filename = f"{model_name}-{POOLING}-whiten(NLI).pkl"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_whiten(output_path, kernel, bias)
    print("Save to {}".format(output_path))

    vector_sentence_path = os.path.join(OUTPUT_DIR, "vctors_sentence.pkl")
    transfered_vec = transform_and_normalize(vectors,kernel, bias)
    sentence_vector = {}
    for s,v in zip(sentence,transfered_vec):
        sentence_vector[s] = v
    with open(vector_sentence_path, 'wb') as handle:
        pickle.dump(sentence_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()