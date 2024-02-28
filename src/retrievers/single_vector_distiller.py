import os
import sys
import pathlib
import logging
import argparse
from os.path import join
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import wandb
import random
import numpy as np
from torch.optim import AdamW
from sklearn.decomposition import PCA
from torch_optimizer import Adafactor
from torch.utils.data import DataLoader
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

from sentence_transformers.losses import MSELoss
from sentence_transformers import util, LoggingHandler
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.datasets import ParallelSentencesDataset

try:
    from src.utils.loggers import WandbLogger
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.loggers import WandbLogger
from src.data.mmarco import MMARCOBiencoderLoader
from src.data.other import SentencesLoader, load_sts_samples
from src.utils.common import (
    set_seed, 
    load_sbert_model,
    count_trainable_parameters, 
    set_xmod_language, 
    prepare_xmod_for_finetuning,
)
from src.utils.SentenceTransformer import (
    MSEEvaluatorCustom, 
    EmbeddingSimilarityEvaluatorCustom, 
    InformationRetrievalEvaluatorCustom,
)


def main(args):
    # Determinism.
    set_seed(args.seed)

    # Output.
    os.makedirs(join(args.output_dir, 'logs'), exist_ok=True)
    out_model_name = f"{args.student_model_name.split('/')[-1]}-from-{args.teacher_model_name.split('/')[-1]}-{args.dataset}"
    out_path = join(args.output_dir, args.dataset, "biencoder", f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{out_model_name}')

    # Loggers.
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])
    logger = WandbLogger(project_name=f"modular-retrievers", run_name=f"biencoder-{out_model_name}", run_config=args, log_dir=join(args.output_dir, 'logs'))

    # Data.
    train_sentences, dev_sentences, _ = SentencesLoader(args.dataset).run()

    # Student.
    student = load_sbert_model(model_name=args.student_model_name, max_seq_length=args.max_seq_length, pooling=args.pooling)
    if student[0].auto_model.__class__.__name__.lower().startswith("xmod") and args.do_train:
        set_xmod_language(model=student[0].auto_model, lang=args.train_lang)
        prepare_xmod_for_finetuning(model=student[0].auto_model)
    args.student_params = count_trainable_parameters(student, verbose=args.do_train)

    # Teacher.
    teacher = load_sbert_model(model_name=args.teacher_model_name, max_seq_length=args.max_seq_length, pooling=args.pooling)
    if student.get_sentence_embedding_dimension() < teacher.get_sentence_embedding_dimension():
        logging.info("Student model has fewer dimensions than the teacher. Computing PCA for down projection...")
        pca = PCA(n_components=student.get_sentence_embedding_dimension())
        pca_embeddings = teacher.encode(train_sentences[0:40000], convert_to_numpy=True)
        pca.fit(pca_embeddings)
        dense = Dense(in_features=teacher.get_sentence_embedding_dimension(), out_features=student.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        teacher.add_module('dense', dense)
    args.teacher_params = count_trainable_parameters(teacher, verbose=args.do_train)
    
    # Dataloader.
    train_data = ParallelSentencesDataset(student_model=student, teacher_model=teacher, batch_size=args.batch_size, use_embedding_cache=True)
    train_data.add_dataset([[sent] for sent in train_sentences], max_sentence_length=None)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)

    # Loss.
    train_loss = MSELoss(model=student)

    # Evaluators.
    dev_evaluator_mse = MSEEvaluatorCustom(name='dev', source_sentences=dev_sentences, target_sentences=dev_sentences, teacher_model=teacher, log_callback=logger.log_eval) # Evaluator that measures the Mean Squared Error (MSE) between the teacher and the student embeddings.
    dev_evaluator_sts = EmbeddingSimilarityEvaluatorCustom.from_input_examples(name='dev', examples=load_sts_samples(), log_callback=logger.log_eval) # Evaluator that measures the performance of student im comparison to teacher on sentence similarity.

    # Train student to imitate teacher.
    total_steps = args.epochs * len(train_dataloader)
    if args.do_train:
        student.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            fp16_amp=args.use_fp16, bf16_amp=args.use_bf16,
            scheduler=args.scheduler, warmup_steps=int(args.warmup_ratio * total_steps),
            optimizer_class=getattr(sys.modules[__name__], args.optimizer), optimizer_params={"lr": args.lr, 'eps': 1e-6}, weight_decay=args.wd, 
            log_every_n_steps=args.log_steps, log_callback=logger.log_training,
            evaluator=SequentialEvaluator([dev_evaluator_sts, dev_evaluator_mse]), evaluation_steps=len(train_dataloader), output_path=out_path,
            checkpoint_path=out_path if args.save_during_training else None, checkpoint_save_steps=len(train_dataloader), checkpoint_save_total_limit=3,
            show_progress_bar=True,
        )
        student.save(f"{out_path}/final")

    # Testing.
    if args.do_test:
        eval_data = MMARCOBiencoderLoader(lang=args.test_lang).run()
        dev_evaluator_ir = InformationRetrievalEvaluatorCustom(
            name='dev', log_callback=logger.log_eval, show_progress_bar=True,
            queries=eval_data['dev_queries'], relevant_docs=eval_data['dev_labels'], corpus=eval_data['corpus'],
            precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500, 1000], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
            score_functions={args.sim: getattr(util, args.sim)},
            corpus_chunk_size=50000, batch_size=args.batch_size,
        )
        if student[0].auto_model.__class__.__name__.lower().startswith("xmod"):
            set_xmod_language(model=student[0].auto_model, lang=args.test_lang)
        student.evaluate(evaluator=dev_evaluator_ir, output_path=out_path, epoch=args.epochs, steps=total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset.
    parser.add_argument("--dataset", type=str, choices=["minipile", "wikipedia", "nli", "msmarco"], help="The collection of sentences used for training.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Models.
    parser.add_argument("--student_model_name", type=str, help="The pretrained (or pre-finetuned) student model ID on Hugging Face.")
    parser.add_argument("--teacher_model_name", type=str, help="The pretrained (or pre-finetuned) teacher model ID on Hugging Face.")
    parser.add_argument("--teacher_is_sbert_model", action="store_true", default=False, help="Whether the teacher model is a trained SentenceTransformer model that can be loaded with the main module.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--pooling", type=str, choices=["mean", "max", "cls"], help="Type of pooling to perform to get a passage representation.")
    parser.add_argument("--sim", type=str, choices=["cos_sim", "dot_score"], help="Similarity function for scoring query-document representation.")
    # Training.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--train_lang", type=str, default="en", help="Language of the trainind dataset and teacher model.")
    parser.add_argument("--batch_size", type=int, help="The batch size per GPU/TPU core/CPU.")
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adafactor", "Shampoo"], help="Type of optimizer to use for training.")
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
    parser.add_argument("--test_lang", type=str, help="Language to perform IR evaluation.")
    args, _ = parser.parse_known_args()
    main(args)
