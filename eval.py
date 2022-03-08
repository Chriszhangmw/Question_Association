import os
import torch
import numpy as np
from tqdm import tqdm
import scipy.stats
from utils import *
import senteval
from prettytable import PrettyTable

MAX_LENGTH = 64
BATCH_SIZE = 256
TEST_PATH = './data/'

MODEL_ZOOS = {

    'BERTbase-whiten-256(target)': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
        'n_components': 256,
    },

    'BERTlarge-whiten-384(target)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'n_components': 384,
    },

    'SBERTbase-nli-whiten-256(target)': {
        'encoder': './model/bert-base-nli-mean-tokens',
        'pooling': 'first_last_avg',
        'n_components': 256,
    },

    'SBERTlarge-nli-whiten-384(target)': {
        'encoder': './model/bert-large-nli-mean-tokens',
        'pooling': 'first_last_avg',
        'n_components': 384
    },

}


def prepare(params, samples):
    samples = [' '.join(sent) if sent != [] else '.' for sent in samples]
    vecs = sents_to_vecs(samples, params['tokenizer'], params['encoder'], \
                         params['pooling'], MAX_LENGTH, verbose=False)
    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :params['n_components']]
    params['whiten'] = (kernel, bias)
    return None


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = []
    for sent in batch:
        vec = sent_to_vec(sent, params['tokenizer'], \
                          params['encoder'], params['pooling'], MAX_LENGTH)
        embeddings.append(vec)
    embeddings = np.vstack(embeddings)
    embeddings = transform_and_normalize(embeddings,
                                         kernel=params['whiten'][0],
                                         bias=params['whiten'][1]
                                         )  # whitening
    return embeddings


def run(model_name, test_path):
    model_config = MODEL_ZOOS[model_name]
    tokenizer, encoder = build_model(model_config['encoder'])

    # Set params for senteval
    params_senteval = {
        'task_path': test_path,
        'usepytorch': True,
        'tokenizer': tokenizer,
        'encoder': encoder,
        'pooling': model_config['pooling'],
        'n_components': model_config['n_components'],
        'batch_size': BATCH_SIZE
    }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        'SICKRelatednessCosin',
        'STSBenchmarkCosin'
    ]
    results = se.eval(transfer_tasks)

    # Show results
    table = PrettyTable(["Task", "Spearman"])
    for task in transfer_tasks:
        if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            metric = results[task]['all']['spearman']['wmean']
        elif task in ['SICKRelatednessCosin', 'STSBenchmarkCosin']:
            metric = results[task]['spearman']
        table.add_row([task, metric])


def run_all_model():
    for model_name in MODEL_ZOOS:
        run(model_name, TEST_PATH)


if __name__ == "__main__":
    # run('BERTbase-whiten-256(target)', TEST_PATH)
    run_all_model()