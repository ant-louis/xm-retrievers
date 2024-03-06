"""
The MS MARCO dataset is a large-scale IR dataset from Microsoft Bing comprising:
- a corpus of 8.8M passages;
- a training set of ~533k queries (with at least one relevant passage);
- a development set of ~101k queries;
- a smaller dev set of 6,980 queries (which is actually used for evaluation in most published works).
The mMARCO dataset is a machine-translated version of MS Marco in 13 different languages.
Link: https://ir-datasets.com/mmarco.html#mmarco/v2/fr/.
"""
import os
import sys
import tqdm
import random
import pathlib
import itertools
import json, gzip, pickle
from collections import defaultdict
from typing import Dict, List, Optional
from os.path import exists, join, basename

import ir_datasets
from torch.utils.data import Dataset
from sentence_transformers import InputExample, util

try:
    from src.utils.common import MMARCO_LANGUAGES
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import MMARCO_LANGUAGES

# https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives
NEGS_MINING_SYSTEMS = [
    'bm25', 
    'msmarco-distilbert-base-tas-b',
    'msmarco-distilbert-base-v3',
    'msmarco-MiniLM-L-6-v3',
    'distilbert-margin_mse-cls-dot-v2',
    'distilbert-margin_mse-cls-dot-v1',
    'distilbert-margin_mse-mean-dot-v1',
    'mpnet-margin_mse-mean-v1',
    'co-condenser-margin_mse-cls-v1',
    'distilbert-margin_mse-mnrl-mean-v1',
    'distilbert-margin_mse-sym_mnrl-mean-v1',
    'distilbert-margin_mse-sym_mnrl-mean-v2',
    'co-condenser-margin_mse-sym_mnrl-mean-v1',
]

class MMARCOColbertLoader:
    def __init__(
        self, 
        lang: str,  # Language in which MS Marco is loaded.
        load_train: Optional[str] = True, # Whether to load the training data.
        load_test: Optional[str] = True, # Whether to load the test data (which actually corresponds to the small dev set in MS Marco).
        train_qrels_type: Optional[str] = 'hard', # Type of training samples to use ("original" or "hard"). The former are official MS MARCO training triples with a single BM25 negative. The latter are custom training samples obtained by mining hard negatives from dense retrievers.
        negs_mining_systems: Optional[str] = '',  # Comma-separated list of systems used for mining hard negatives (if train_qrels_type == 'hard').
        negs_per_query: Optional[int] = 39,  # Number of hard negatives to sample per query (if train_qrels_type == 'hard').
        num_sampling_rounds: Optional[int] = 1,  # Number of times each query is iterated over with a different set of randomly sampled negative passages. Given 532,761 original training queries, 5 rounds would create 2,663,805 training tuples with different randomly sampled hard negatives.
        ce_score_margin: Optional[float] = 3.0,  # Margin for the cross-encoder score between negative and positive passages.
        data_folder: Optional[str] = 'data/mmarco',  # Folder in which to save the downloaded datasets.
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        assert train_qrels_type in ["original", "hard"], f"Unkwown type of training qrels. Please choose between 'original' and 'hard'."
        if negs_mining_systems:
            assert all(syst in NEGS_MINING_SYSTEMS for syst in negs_mining_systems.split(',')), f"Unknown hard negative mining system."
        self.lang = lang
        self.load_train = load_train
        self.load_test = load_test
        self.train_qrels_type = train_qrels_type
        self.negs_mining_systems = negs_mining_systems
        self.negs_per_query = negs_per_query
        self.num_sampling_rounds = num_sampling_rounds
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder

    def run(self):
        data_filepaths = {}

        # Load collection of passages.
        url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/{MMARCO_LANGUAGES.get(self.lang)[0]}_collection.tsv'
        data_filepaths['collection'] = self.download_if_not_exists(url)

        if self.load_test:
            # Load test queries. 
            url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/{MMARCO_LANGUAGES.get(self.lang)[0]}_queries.dev.small.tsv'
            data_filepaths['test_queries'] = self.download_if_not_exists(url)

            # Load test qrels. 
            url = 'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.small.tsv'
            data_filepaths['test_qrels'] = self.download_if_not_exists(url)

        if self.load_train:
            # Load training queries. 
            url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/{MMARCO_LANGUAGES.get(self.lang)[0]}_queries.train.tsv'
            data_filepaths['train_queries'] = self.download_if_not_exists(url)

            # Load training qrels. 
            if self.train_qrels_type == "original":
                url = 'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/triples.train.ids.small.tsv'
                save_path = self.download_if_not_exists(url)
                data_filepaths['train_triples'] = self.tsv_to_jsonl(save_path)
            else:
                examples = 502939 * self.num_sampling_rounds # MS MARCO official training set contains 808,731 queries yet only 502,939 have associated labels.
                training_tuples_filepath = join(self.data_folder, f'tuples.train.scores-ids.{self.negs_per_query+1}way.{examples/1e6:.1f}M.jsonl')
                if not exists(training_tuples_filepath):
                    num_training_examples = 0

                    # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
                    ce_scores_file = self.download_if_not_exists(url)
                    with gzip.open(ce_scores_file, 'rb') as fIn:
                        ce_scores = pickle.load(fIn)

                    # Load hard negatives mined from BM25 and 12 different dense retrievers.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
                    hard_negatives_filepath = self.download_if_not_exists(url)
                    with gzip.open(hard_negatives_filepath, 'rt') as fIn, open(training_tuples_filepath, 'w') as fOut:
                        for round_idx in tqdm.tqdm(range(self.num_sampling_rounds), desc='Sampling round'):
                            fIn.seek(0)
                            random.seed(42 + round_idx)
                            for line in tqdm.tqdm(fIn):
                                # Load the training sample: {"qid": ..., "pos": [...], "neg": {"bm25": [...], "msmarco-MiniLM-L-6-v3": [...], ...}}
                                data = json.loads(line)
                                qid = data['qid']
                                pos_pids = data['pos']
                                if len(pos_pids) == 0:
                                    continue

                                # Set the CE threshold as the minimum positive score minus a margin.
                                pos_min_ce_score = min([ce_scores[qid][pid] for pid in pos_pids])
                                ce_score_threshold = pos_min_ce_score - self.ce_score_margin

                                # Sample one positive passage and its associated CE scores.
                                sampled_pos_pid = random.choice(pos_pids)
                                sampled_pos_score = ce_scores[qid][sampled_pos_pid]
                                sampled_pos_tuple = [[sampled_pos_pid, sampled_pos_score]]

                                # Sample N hard negatives and their CE scores.
                                neg_pids = []
                                neg_systems = self.negs_mining_systems.split(",") if self.negs_mining_systems else list(data['neg'].keys())
                                for system_name in neg_systems:
                                    neg_pids.extend(data['neg'][system_name])
                                filtered_neg_pids = [pid for pid in list(set(neg_pids)) if ce_scores[qid][pid] <= ce_score_threshold]
                                sampled_neg_pids = random.sample(filtered_neg_pids, min(self.negs_per_query, len(filtered_neg_pids)))
                                sampled_neg_scores = [ce_scores[qid][pid] for pid in sampled_neg_pids]
                                sampled_neg_tuples = [list(pair) for pair in zip(sampled_neg_pids, sampled_neg_scores)]

                                if len(sampled_neg_tuples) == self.negs_per_query:
                                    sample = [qid] + sampled_pos_tuple + sampled_neg_tuples
                                    fOut.write(json.dumps(sample) + '\n')
                                    num_training_examples += 1

                    print(f"#> Number of training examples created: {num_training_examples}.")
        
                data_filepaths['train_triples'] = training_tuples_filepath
    
        return data_filepaths

    def download_if_not_exists(self, file_url: str):
        save_path = join(self.data_folder, basename(file_url))
        if not exists(save_path):
            util.http_get(file_url, save_path)
        return save_path

    def tsv_to_jsonl(self, tsv_filename: str):
        jsonl_filename = tsv_filename.replace('.tsv', '.jsonl')
        with open(tsv_filename, 'r') as fIn, open(jsonl_filename, 'w') as fOut:
            for line in tqdm(fIn, desc="Converting"):
                parts = [int(pid) for pid in line.strip().split('\t')]
                fOut.write(json.dumps(parts) + '\n')
        return jsonl_filename


class MMARCOCrossencoderLoader:
    """Source: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
    """
    def __init__(
        self, 
        lang: str, #Language in which MS Marco is loaded.
        data_folder: str = 'data/mmarco', #Folder in which to save the downloaded datasets.
        num_dev_queries: int = 500, #Number of random queries from the train set to use for evaluation during training.
        num_max_dev_negatives: int = 200, #Maximum number of negatives to use per dev query.
        pos_neg_ration: int = 4, #Positive-to-negative ratio in our training setup for the binary label task. For 1 positive sample (label 1) we include 4 negative samples (label 0) by sampling from the (query, positive sample, negative sample) triplets provided by MS Marco.
        max_train_samples: int = 1e6, #Maximal number of training samples we want to use.
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        self.lang = lang.replace('nl', 'dt')
        self.collection = "msmarco-passage" if lang == "en" else f"mmarco/v2/{self.lang}"
        self.data_folder = data_folder
        self.num_dev_queries = num_dev_queries
        self.num_max_dev_negatives = num_max_dev_negatives
        self.pos_neg_ration = pos_neg_ration
        self.max_train_samples = max_train_samples

    def run(self):
        train = ir_datasets.load(f"{self.collection}/train")
        corpus = {d.doc_id: d.text for d in train.docs_iter()}
        train_queries = {q.query_id: q.text for q in train.queries_iter()}
        dev_samples = self.build_dev_set(train_queries, corpus)
        train_samples = self.build_training_set(train_queries, corpus, dev_samples)
        return {'train': train_samples, 'dev': dev_samples}

    def build_dev_set(self, queries: Dict[int, str], corpus: Dict[int, str]):
        train_eval_filepath = join(self.data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
        if not exists(train_eval_filepath):
            util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)
        dev_samples = {}
        with gzip.open(train_eval_filepath, 'rt') as fIn:
            for line in fIn:
                qid, pos_id, neg_id = line.strip().split()
                if qid not in dev_samples and len(dev_samples) < self.num_dev_queries:
                    dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}
                if qid in dev_samples:
                    dev_samples[qid]['positive'].add(corpus[pos_id])
                    if len(dev_samples[qid]['negative']) < self.num_max_dev_negatives:
                        dev_samples[qid]['negative'].add(corpus[neg_id])
        return dev_samples

    def build_training_set(self, queries: Dict[int, str], corpus: Dict[int, str], dev_samples):
        train_filepath = join(self.data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
        if not exists(train_filepath):
            util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)
        cnt = 0
        train_samples = []
        with gzip.open(train_filepath, 'rt') as fIn:
            for line in tqdm.tqdm(fIn, unit_scale=True):
                qid, pos_id, neg_id = line.strip().split()
                if qid in dev_samples:
                    continue
                query = queries[qid]
                if (cnt % (self.pos_neg_ration + 1)) == 0:
                    passage, label = corpus[pos_id], 1
                else:
                    passage, label = corpus[neg_id], 0
                train_samples.append(InputExample(texts=[query, passage], label=label))
                cnt += 1
                if cnt >= self.max_train_samples:
                    break
        return train_samples



class MMARCOBiencoderLoader:
    """Source: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
    """
    def __init__(
        self, 
        lang: str,  # Language in which MS Marco is loaded.
        negs_mining_systems: Optional[str] = '',  # Comma-separated list of systems used for mining hard negatives.
        num_negs_per_system: Optional[int] = 5,  # Number of negatives to use per system.
        ce_score_margin: Optional[float] = 3.0,  # Margin for the cross-encoder score between negative and positive passages.
        data_folder: Optional[str] = 'data/mmarco',  # Folder in which to save the downloaded datasets.
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        if negs_mining_systems:
            assert all(syst in NEGS_MINING_SYSTEMS for syst in negs_mining_systems.split(',')), f"Unknown hard negative mining system."
        self.lang = lang.replace('nl', 'dt')
        self.collection = "msmarco-passage" if lang == "en" else f"mmarco/v2/{self.lang}"
        self.negs_mining_systems = negs_mining_systems
        self.num_negs_per_system = num_negs_per_system
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder

    def run(self):
        # Load train set.
        train = ir_datasets.load(f"{self.collection}/train")
        corpus = {int(d.doc_id): d.text for d in train.docs_iter()}
        train_queries = {int(q.query_id): q.text for q in train.queries_iter()}
        train_samples = self.build_training_set(train_queries)
        train_dataset = MMARCODataset(train_samples, corpus)
        
        # Load dev set.
        dev = ir_datasets.load(f"{self.collection}/dev/small")
        dev_queries = {int(q.query_id): q.text for q in dev.queries_iter()}
        dev_labels = {}
        for sample in dev.qrels_iter():
            dev_labels.setdefault(int(sample.query_id), []).append(int(sample.doc_id))
        return {'train': train_dataset, 'corpus': corpus, 'dev_queries': dev_queries, 'dev_labels': dev_labels}

    def build_training_set(self, queries: Dict[int, str]):
        # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
        ce_scores_file = join(self.data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
        if not exists(ce_scores_file):
            util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)
        with gzip.open(ce_scores_file, 'rb') as fIn:
            ce_scores = pickle.load(fIn)

        # Load hard negatives mined from BM25 and 12 different dense retrievers.
        hard_negatives_filepath = join(self.data_folder, 'msmarco-hard-negatives.jsonl.gz')
        if not exists(hard_negatives_filepath):
            util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)

        train_samples = {}
        with gzip.open(hard_negatives_filepath, 'rt') as fIn:
            for line in tqdm.tqdm(fIn):
                # Load the traininsg sample: {"qid": ..., "pos": [...], "neg": {"bm25": [...], "msmarco-MiniLM-L-6-v3": [...], ...}}
                data = json.loads(line)
                qid = data['qid']

                # Get positive passages and their CE scores.
                pos_pids = data['pos']
                if len(pos_pids) == 0:
                    continue
                
                # Set the CE threshold as the...
                pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
                ce_score_threshold = pos_min_ce_score - self.ce_score_margin

                # Sample the hard negatives.
                neg_pids = set()
                if self.negs_mining_systems:
                    neg_systems = self.negs_mining_systems.split(",")
                else:
                    neg_systems = list(data['neg'].keys())
                for system_name in neg_systems:
                    if system_name not in data['neg']:
                        continue
                    system_negs = data['neg'][system_name]
                    negs_added = 0
                    for pid in system_negs:
                        if ce_scores[qid][pid] > ce_score_threshold:
                            continue
                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= self.num_negs_per_system:
                                break
                if len(pos_pids) > 0 and len(neg_pids) > 0:
                    train_samples[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
        del ce_scores
        return train_samples


class MMARCODataset(Dataset):
    """
    Source: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
    """
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0) # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0) # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)
