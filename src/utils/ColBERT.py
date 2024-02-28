import os
import json
import time
from tqdm import tqdm
from os.path import join, dirname
from collections import defaultdict

import torch
import random
import numpy as np
from langdetect import detect
from torch.optim import AdamW
import torch.multiprocessing as mp
from transformers import get_linear_schedule_with_warmup

from colbert import Trainer, Indexer, Searcher
from colbert.infra import ColBERTConfig, Run
from colbert.infra.launcher import Launcher, print_memory_stats

from colbert.training.training import set_bert_grad
from colbert.training.utils import manage_checkpoints
from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.rerank_batcher import RerankBatcher

from colbert.parameters import DEVICE
from colbert.utils.amp import MixedPrecisionManager
from colbert.utils.utils import print_message, file_tqdm

from colbert.modeling.colbert import ColBERT
from colbert.modeling.checkpoint import Checkpoint

from colbert.data.collection import Collection
from colbert.search.index_storage import IndexScorer
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.collection_indexer import CollectionIndexer


from .loggers import WandbLogger
from .common import set_seed, set_xmod_language, prepare_xmod_for_finetuning


#-----------------------------------------------------------------------------------------------------------------#
#                                               TRAINER
#-----------------------------------------------------------------------------------------------------------------#
class CustomTrainer(Trainer):
    def train(self, checkpoint='facebook/xmod-base'):
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)
        launcher = Launcher(custom_train)
        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)


def custom_train(config: ColBERTConfig, triples, queries=None, collection=None):
    if config.rank < 1:
        config.help()

    os.makedirs(join(config.root, 'logs'), exist_ok=True)
    logger = WandbLogger(
        project_name=config.experiment,
        run_name=f"colbert-{config.checkpoint}",
        run_config=config, 
        log_dir=join(config.root, 'logs'),
    )
    set_seed(42)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks
    print("#> Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is None:
        raise NotImplementedError()
    reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    
    colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    if colbert.bert.__class__.__name__.lower().startswith("xmod"):
        language = detect(reader.collection.__getitem__(0))
        print(f"#> Training an X-MOD model in {language}.")
        set_xmod_language(colbert.bert, lang=language)
        prepare_xmod_for_finetuning(colbert.bert)
    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert, 
        device_ids=[config.rank],
        output_device=config.rank,
        find_unused_parameters=True,
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-6)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
        )

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None

    for batch_idx, BatchSteps in zip(tqdm(range(0, config.maxsteps), desc="Step"), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = torch.nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    loss += ib_loss

                loss = loss / config.accumsteps

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = 0.999 * train_loss + 0.001 * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            logger.log_training(0, 0, batch_idx, scheduler.get_last_lr()[0], this_batch_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)
        return ckpt_path


#-----------------------------------------------------------------------------------------------------------------#
#                                               INDEXER
#-----------------------------------------------------------------------------------------------------------------#
class CustomIndexer(Indexer):
    def __launch(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]
        launcher = Launcher(custom_encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues, self.verbose)

def custom_encode(config, collection, shared_lists, shared_queues, verbose: int = 3):
    encoder = CustomCollectionIndexer(config=config, collection=collection, verbose=verbose)
    encoder.run(shared_lists)

class CustomCollectionIndexer(CollectionIndexer):
    def __init__(self, config: ColBERTConfig, collection, verbose=2):
        self.verbose = verbose
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks
        self.use_gpu = self.config.total_visible_gpus > 0
        if self.config.rank == 0 and self.verbose > 1:
            self.config.help()
        self.collection = Collection.cast(collection)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config)
        if self.checkpoint.bert.__class__.__name__.lower().startswith("xmod"):
            language = detect(self.collection.__getitem__(0))
            Run().print_main(f"#> Setting X-MOD language adapters to {language}.")
            set_xmod_language(self.checkpoint.bert, lang=language)
        if self.use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)
        print_memory_stats(f'RANK:{self.rank}')


#-----------------------------------------------------------------------------------------------------------------#
#                                               SEARCHER
#-----------------------------------------------------------------------------------------------------------------#
class CustomSearcher(Searcher):
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, verbose:int = 3):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=self.verbose)
        if self.checkpoint.bert.__class__.__name__.lower().startswith("xmod"):
            language = detect(self.collection.__getitem__(0))
            print_message(f"#> Setting X-MOD language adapters to {language}.")
            set_xmod_language(self.checkpoint.bert, lang=language)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)
        print_memory_stats()


#-----------------------------------------------------------------------------------------------------------------#
#                                          MS MARCO EVALUATION
#-----------------------------------------------------------------------------------------------------------------#
def msmarco_evaluation(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [10, 50, 100, 200, 500, 1000]}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1
            qid2positives[qid].append(pid)

    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)
            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None
            qid2ranking[qid].append((rank, pid, score))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))
    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
        print_message(f"#> {num_judged_queries} != {num_ranked_queries}")

    print_message(f"#> Computing MRR@10 for {num_judged_queries} queries.")
    for qid in tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]
        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed
            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed
            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    results = {}
    mrr_10_sum = sum(qid2mrr.values())
    results['mrr_10'] = mrr_10_sum / num_judged_queries
    print_message(f"#> MRR@10 = {results['mrr_10']}")
    #print_message(f"#> MRR@10 (only for ranked queries) = {mrr_10_sum / num_ranked_queries}")

    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)
        metric_sum = sum(qid2recall[depth].values())
        results[f"recall@{depth}"] = metric_sum / num_judged_queries
        print_message(f"#> Recall@{depth} = {results[f'recall@{depth}']}")
        #print_message(f"#> Recall@{depth} (only for ranked queries) = {metric_sum / num_ranked_queries}")

    os.makedirs(dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output
        print_message(f"#> Writing annotations to {args.output} ..")
        with open(args.output, 'w') as f:
            for qid in tqdm(qid2positives):
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]
                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)
                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)
