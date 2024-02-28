import os
import sys
import pathlib
import logging
import argparse
from os.path import join
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import torch
from torch import nn
from torch.optim import AdamW
from torch_optimizer import Adafactor
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import wandb
import random
import numpy as np
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

from sentence_transformers import util, LoggingHandler
from sentence_transformers.losses import MultipleNegativesRankingLoss

try:
    from src.data.mmarco import MMARCOBiencoderLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.mmarco import MMARCOBiencoderLoader
from src.utils.loggers import WandbLogger
from src.utils.common import (
    set_seed, 
    load_sbert_model,
    count_trainable_parameters, 
    set_xmod_language, 
    prepare_xmod_for_finetuning,
)
from src.utils.SentenceTransformer import InformationRetrievalEvaluatorCustom


def main(args):
    # Determinism.
    set_seed(args.seed)

    # Output.
    os.makedirs(join(args.output_dir, 'logs'), exist_ok=True)
    out_model_name = f"{args.model_name.split('/')[-1]}-{args.dataset}"
    out_path = join(args.output_dir, args.dataset, "biencoder", f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{out_model_name}')

    # Loggers.
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])
    logger = WandbLogger(project_name=f"modular-retrievers", run_name=f"biencoder-{out_model_name}", run_config=args, log_dir=join(args.output_dir, 'logs'))
    
    # Model.
    logging.info("Creating new biencoder model...")
    model = load_sbert_model(model_name=args.model_name, max_seq_length=args.max_seq_length, pooling=args.pooling)
    if model[0].auto_model.__class__.__name__.lower().startswith("xmod"):
        set_xmod_language(model=model[0].auto_model, lang=args.language)
        if args.do_train:
            prepare_xmod_for_finetuning(model=model[0].auto_model)
    args.model_params = count_trainable_parameters(model, verbose=args.do_train)

    # Loss.
    loss = MultipleNegativesRankingLoss(model=model, similarity_fct=getattr(util, args.sim))

    # Data.
    if args.dataset == 'mmarco':
        data = MMARCOBiencoderLoader(lang=args.language).run()
    else:
        raise ValueError('Invalid dataset specified')
    train_dataloader = DataLoader(data['train'], shuffle=True, batch_size=args.train_batch_size)

    # Training.
    total_steps = args.epochs * len(train_dataloader)
    if args.do_train:
        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=args.epochs,
            fp16_amp=args.use_fp16, bf16_amp=args.use_bf16,
            scheduler=args.scheduler, warmup_steps=int(args.warmup_ratio * total_steps),
            optimizer_class=getattr(sys.modules[__name__], args.optimizer), optimizer_params={"lr": args.lr}, weight_decay=args.wd, 
            log_every_n_steps=args.log_steps, log_callback=logger.log_training,
            evaluator=None, evaluation_steps=0, output_path=out_path,
            checkpoint_path=out_path if args.save_during_training else None, checkpoint_save_steps=len(train_dataloader), checkpoint_save_total_limit=3,
            show_progress_bar=True,
        )
        model.save(f"{out_path}/final")

    # Testing.
    if args.do_test:
        dev_evaluator = InformationRetrievalEvaluatorCustom(
            name='dev', log_callback=logger.log_eval, show_progress_bar=True,
            queries=eval_data['dev_queries'], relevant_docs=eval_data['dev_labels'], corpus=eval_data['corpus'],
            precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
            score_functions={args.sim: getattr(util, args.sim)},
            corpus_chunk_size=50000, batch_size=args.batch_size,
        )
        model.evaluate(evaluator=dev_evaluator, output_path=out_path, epoch=args.epochs, steps=total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data.
    parser.add_argument("--dataset", type=str, choices=["mmarco"], help="The dataset to perform training and evaluation on.")
    parser.add_argument("--language", type=str, choices=['ar', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'nl', 'pt', 'ru', 'vi', 'zh'], help="Target language for training on mMARCO.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Model.
    parser.add_argument("--model_name", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--pooling", type=str, choices=["mean", "max", "cls"], help="Type of pooling to perform to get a passage representation.")
    parser.add_argument("--sim", type=str, choices=["cos_sim", "dot_score"], help="Similarity function for scoring query-document representation.")
    # Training.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adafactor"], help="Type of optimizer to use for training.")
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--wd", type=float, help="The weight decay to apply (if not zero) to all layers in AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, choices=["constantlr", "warmupconstant", "warmuplinear", "warmupcosine", "warmupcosinewithhardrestarts"], help="Type of learning rate scheduler to use for training.")
    parser.add_argument("--warmup_ratio", type=float, help="Ratio of total training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--use_fp16", action="store_true", default=False, help="Whether to use mixed precision during training.")
    parser.add_argument("--use_bf16", action="store_true", default=False, help=".")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--save_during_training", action="store_true", default=False, help="Wether to save model checkpoints during training.")
    parser.add_argument("--log_steps", type=int, help="Log every k training steps.")
    # Evaluation.
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform test evaluation after training.")
    args, _ = parser.parse_known_args()
    main(args)
