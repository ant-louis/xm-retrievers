import sys
import json
import math
import torch
import shutil
import pathlib
import argparse
from tqdm import tqdm
from os.path import join
from copy import deepcopy
from datetime import datetime

from colbert.data import Queries, Collection
from colbert.infra import Run, RunConfig, ColBERTConfig

try:
    from src.data.mmarco import MMARCOColbertLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.mmarco import MMARCOColbertLoader
from src.data.mrtydi import MrTydiColbertLoader
from src.utils.ColBERT import CustomTrainer, CustomIndexer, CustomSearcher, msmarco_evaluation


def main(args):
    if args.dataset == 'mmarco':
        data_filepaths = MMARCOColbertLoader(
            lang=args.language,
            load_train=args.do_train,
            load_test=args.do_test,
            train_qrels_type='hard',
            negs_per_query=args.nway-1,
            num_sampling_rounds=math.ceil((args.maxsteps * args.bsize) / 502939),
            data_folder=join(args.data_dir, args.dataset),
        ).run()
    elif args.dataset == 'mrtydi':
        data_filepaths = MrTydiColbertLoader(
            lang=args.language,
            load_train=args.do_train,
            load_test=args.do_test,
            data_folder=join(args.data_dir, args.dataset),
        ).run()
    else:
        raise ValueError("Dataset not supported.")

    run_kwargs = {
        'rank': 0,
        'amp': True,
        'nranks': torch.cuda.device_count(),
        'root': join(args.output_dir, args.dataset),
        'experiment': 'modular-retrievers',
        'name': f'{datetime.now().strftime("%Y-%m-%d_%H.%M")}-{args.model_name.replace("/", "-")}-{args.dataset}-{args.language}',
    }
    model_kwargs = deepcopy(vars(args))
    model_kwargs['checkpoint'] = args.model_name
    for k in {'dataset', 'language', 'data_dir', 'output_dir', 'do_train', 'do_test'}:
        model_kwargs.pop(k, None)

    if args.do_train:
        with Run().context(RunConfig(**run_kwargs)):
            trainer = CustomTrainer(
                config=ColBERTConfig(**model_kwargs),
                triples=data_filepaths['train_triples'],
                queries=data_filepaths['train_queries'],
                collection=data_filepaths['collection'],
            )
            model_kwargs['checkpoint'] = trainer.train(checkpoint=args.model_name)

    if args.do_test:
        if 'checkpoints' in model_kwargs['checkpoint']:
            model_path, ckpt_name = model_kwargs['checkpoint'].split('/checkpoints/', 1)
        else:
            model_path, ckpt_name = join(run_kwargs['root'], run_kwargs['experiment'], model_kwargs['checkpoint'].split('/')[-1]), ""
        run_kwargs['index_root'] = join(model_path, 'indexes', ckpt_name)
        with Run().context(RunConfig(**run_kwargs)):
            indexer = CustomIndexer(checkpoint=model_kwargs['checkpoint'], config=ColBERTConfig(**model_kwargs))
            indexer.index(name=f"{args.dataset}-{args.language}.index", collection=data_filepaths['collection'], overwrite='reuse')

        with Run().context(RunConfig(**run_kwargs)):
            queries = Queries(data_filepaths['test_queries'])
            searcher = CustomSearcher(index=f"{args.dataset}-{args.language}.index", config=ColBERTConfig(**model_kwargs))
            ranking = searcher.search_all(queries, k=1000)
            results_path = ranking.save(f"{args.dataset}-{args.language}-ranking.tsv")
            msmarco_evaluation(argparse.Namespace(
                ranking=results_path, 
                qrels=data_filepaths['test_qrels'], 
                annotate=False, 
                output_filepath=join(model_path, 'evaluation', ckpt_name, f"results_{args.dataset}-{args.language}.json"),
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Document settings.
    parser.add_argument("--dim", type=int,
        help="Dimensionality of the embeddings."
    )
    parser.add_argument("--doc_maxlen", type=int,
        help="Maximum length at which the passages will be truncated."
    )
    parser.add_argument("--mask_punctuation", action="store_true", default=False, 
        help="Whether to mask punctuation tokens."
    )
    # Query settings.
    parser.add_argument("--query_maxlen", type=int,
        help="Maximum length at which the queries will be truncated."
    )
    parser.add_argument("--attend_to_mask_tokens", action="store_true", default=False, 
        help="Whether to attend to mask tokens."
    )
    # Training settings.
    parser.add_argument("--do_train", action="store_true", default=False, 
        help="Wether to perform training."
    )
    parser.add_argument("--model_name", type=str,
        help="The model checkpoint for weights initialization."
    )
    parser.add_argument("--similarity", type=str, choices=["cosine", "l2"],
        help="Similarity function for scoring query-document representation."
    )
    parser.add_argument("--bsize", type=int,
        help="The batch size per GPU/TPU core/CPU for training."
    )
    parser.add_argument("--accumsteps", type=int,
        help="The number of accumulation steps before performing a backward/update pass."
    )
    parser.add_argument("--lr", type=float,
        help="The initial learning rate for AdamW optimizer."
    )
    parser.add_argument("--maxsteps", type=int,
        help="The total number of training steps to perform."
    )
    parser.add_argument("--warmup", type=int, default=None,
        help="Number of warmup steps for the learning rate scheduler."
    )
    parser.add_argument("--nway", type=int,
        help="Number of passages/documents to compare the query with. Usually, 1 positive passage + k negative passages."
    )
    parser.add_argument("--use_ib_negatives", action="store_true", default=False, 
        help="Whether to use in-batch negatives during training."
    )
    parser.add_argument("--distillation_alpha", type=float,
        help="""Scaling parameter of the target scores when optimizing with KL-divergence loss.
        A higher value increases the differences between the target scores before applying softmax, leading to a more polarized probability distribution.
        A lower value makes the target scores more similar, leading to a softer probability distribution."""
    )
    parser.add_argument("--ignore_scores", action="store_true", default=False, 
        help="""Whether to ignore scores provided for the n-way tuples. If so, pairwise softmax cross-entropy loss will be applied.
        Otherwise, KL-divergence loss between the target and log scores will be applied."""
    )
    # Index settings (for evaluation).
    parser.add_argument("--do_test", action="store_true", default=False, 
        help="Wether to perform test evaluation after training."
    )
    parser.add_argument("--nbits", type=int,
        help="Number of bits for encoding each dimension."
    )
    parser.add_argument("--kmeans_niters", type=int,
        help="Number of iterations for k-means clustering. 4 is a good and fast default. Consider larger numbers for small datasets."
    )
    # Search settings.
    #parser.add_argument("--ncells", type=int, default=1, 
    #   help="Number of cells for the IVF index."
    # )
    #parser.add_argument("--centroid_score_threshold", type=float,  default=0.5,
    #   help="Threshold for the centroid score."
    # )
    # Data settings.
    parser.add_argument("--dataset", type=str, choices=["mmarco", "mrtydi"],
        help="Dataset to use for training."
    )
    parser.add_argument("--language", type=str,
        help="Target language for training on mMARCO."
    )
    parser.add_argument("--data_dir", type=str,
        help="Folder containing the training data."
    )
    parser.add_argument("--output_dir", type=str,
        help="Folder to save checkpoints, logs, and evaluation results."
    )
    args, _ = parser.parse_known_args()
    main(args)
