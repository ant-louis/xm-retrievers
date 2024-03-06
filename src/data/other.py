import csv
import gzip
import json
import tqdm
import pickle
import random
from os.path import exists, join
from typing import Dict, Optional

import ir_datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from sentence_transformers import util, InputExample


def load_sts_samples():
    sts_dataset_path = 'data/stsbenchmark.tsv.gz'
    if not exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        dev_samples = []
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize in [0,1]
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    return dev_samples


class SentencesLoader:
    """ Class to load raw sentences from different datasets.
    """
    def __init__(self, dataset: str):
        assert dataset in ["minipile", "wikipedia", "nli", "msmarco"]
        self.dataset = dataset

    def run(self):
        if self.dataset == "wikipedia":
            # 7,871,825 sentences
            wikipedia_dataset_path = 'data/wikipedia-en-sentences.txt.gz'
            if not exists(wikipedia_dataset_path):
                util.http_get('https://sbert.net/datasets/wikipedia-en-sentences.txt.gz', wikipedia_dataset_path)
            with gzip.open(wikipedia_dataset_path, 'rt', encoding='utf8') as fIn:
                wikipeda_sentences = [line.strip() for line in fIn]
            dev_sentences = wikipeda_sentences[0:5000]
            train_sentences = wikipeda_sentences[5000:]
            return train_sentences, dev_sentences, None
        
        elif self.dataset == "nli":
            # 1,172,368 sentences
            nli_dataset_path = 'data/AllNLI.tsv.gz'
            if not exists(nli_dataset_path):
                util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
            with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
                train_sentences = set()
                dev_sentences = set()
                for row in reader:
                    if row['split'] == 'dev':
                        dev_sentences.add(row['sentence1'])
                        dev_sentences.add(row['sentence2'])
                    else:
                        train_sentences.add(row['sentence1'])
                        train_sentences.add(row['sentence2'])
            return list(train_sentences), list(dev_sentences)[0:5000], None

        elif self.dataset == "minipile":
            # 1,000,000 sentences
            data = load_dataset('JeanKaddour/minipile')
            train_sentences = data['train']['text']
            dev_sentences = data['validation']['text']
            test_sentences = data['test']['text']
            return train_sentences, dev_sentences, test_sentences

        elif self.dataset == "msmarco":
            # 8,840,000 sentences
            data = load_dataset('BeIR/msmarco', name='corpus', split='corpus')
            dev_sentences = data['text'][0:5000]
            train_sentences = data['text'][5000:]
            return train_sentences, dev_sentences, None

        return None, None, None
