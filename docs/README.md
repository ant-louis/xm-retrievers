# Documentation

### Setup

This repository is tested on Python 3.10+. First, you should install a virtual environment:

```bash
python3 -m venv .venv/modular
source .venv/modular/bin/activate
```

Then, you can install all dependencies:

```bash
pip install -r requirements.txt
```

## ColBERT-XM: a modular dense multi-vector bi-encoder

ColBERT-XM is an [XMOD](https://arxiv.org/abs/2205.06266)-based multi-vector representation model that uses the MaxSim-based late interaction scoring mechanism for relevance matching, as presented in [ColBERT](https://arxiv.org/abs/2004.12832).

### Training

Training can be performed by running:

```bash
bash scripts/run_multi_vector_biencoder.sh
```

after having set the following bash variables in the script:

* `DO_TRAIN="--do_train"`: setup training mode;
* `DO_TEST=""`: deactivate testing mode;
* `DATA="mmarco"`: training uses the [msmarco-hard-negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives) dataset;
* `LANG="en"`: training is performed on the English samples of mMARCO.

Other relevant variables for training include:

* `TRAINING_TYPE` (str, default="v1.5"): either "v1" as in [ColBERTv1](https://arxiv.org/abs/2004.12832), "v2" as in [ColBERTv2](https://arxiv.org/abs/2112.01488), or "v1.5" (same pairwise softmax cross-entropy loss as ColBERTv1 + in-batch sampled softmax cross-entropy loss of ColBERTv2);
* `MODEL` (str, default="[facebook/xmod-base](https://huggingface.co/facebook/xmod-base)"): checkpoint ID from Hugging Face;
* `DIM` (int, default=128): dimension of the dense vectors;
* `SIM` (str, default="cosine"): similarity function used to compare the token-level contextualized embeddings (either "cosine" or "l2");
* `DOC_MAXLEN` (int, default=256): maximum length of the documents;
* `MASK_PUNCT_IN_DOCS="--mask_punctuation"`: whether to mask punctuation in the documents;
* `QUERY_MAXLEN` (int, default=32): maximum length of the queries;
* `ATTEND_MASK_TOKENS_IN_QUERIES="--attend_to_mask_tokens"`: whether to augment queries with mask tokens up to the query maximum length;
* `TOTAL_STEPS` (int): total number of training steps;
* `WARMUP_STEPS` (int): number of warmup steps;
* `LR` (float): learning rate;
* `BATCH_SIZE` (int): training batch size;
* `ACC_STEPS` (int): number of accumulation steps;
* `NEGS_PER_QUERY` (int): number of negatives per query (only used in ColBERTv2);
* `USE_INBATCH_NEGS` (str): whether to use in-batch negatives (relevant for v1.5 and v2);
* `IGNORE_PROVIDED_SCORES_IF_ANY` (str): whether to ignore provided cross-encoder relevance scores if any to not use KL-divergence loss (relevant for v1);
* `DISTIL_ALPHA` (float): distillation alpha for matching cross-encoder scores in KL loss (relevant for v2).

### Evaluation

Evaluation can be performed using the same script after setting the following bash variables:

* `DO_TEST="--do_test"`: setup testing mode;
* `DO_TRAIN=""`: deactivate training mode;
* `DATA="mmmarco"` or `DATA="mrtydi"`: evaluation can be performed on both the [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) and [Mr. TyDI](https://huggingface.co/datasets/castorini/mr-tydi) datasets;
* `LANG="xx"`: language code to evaluate on. For mMARCO, choose between: ("en" "fr" "ar" "de" "es" "hi" "id" "it" "ja" "nl" "pt" "ru" "vi" "zh"). For Mr. TyDi, choose between: ("ar" "bn" "en" "fi" "id" "ja" "ko" "ru" "sw" "te").

Other relevant variables for evaluation include:

* `NBITS` (int, default=2): number of bits to use for quantization of the dense residual vectors during indexing;
* `KMEANS_ITERS` (int, default=4): number of iterations for the k-means clustering algorithm.

## DPR-XM: a modular dense single-vector bi-encoder

DPR-XM is a siamese [XMOD](https://arxiv.org/abs/2205.06266)-based single-vector representation model that uses a conventional single-vector similarity function for relevance matching, as presented in [DPR](https://arxiv.org/abs/2004.04906).

### Standard training

Training can be performed by running:

```bash
bash scripts/run_single_vector_biencoder.sh
```

after having set the following bash variables in the script:

* `DO_TRAIN="--do_train"`: setup training mode;
* `DO_TEST=""`: deactivate testing mode;
* `DATA="mmarco"`: training uses the [msmarco-hard-negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives) dataset;
* `LANG="en"`: training is performed on the English samples of mMARCO;

Other relevant variables for training include:

* `MODEL` (str, default="[facebook/xmod-base](https://huggingface.co/facebook/xmod-base)"): checkpoint ID from Hugging Face;
* `MAX_SEQ_LEN` (int, default=128): maximum input length for both the queries and documents (longer inputs are truncated);
* `POOL` (str, default="mean"): pooling strategy to distil a global sequence representation from the token-level contextualized embeddings (either "mean", "max", or "cls");
* `SIM` (str, default="cos_sim"): similarity function used to compare the query and document representations (either "cos_sim" or "dot_product");
* `BATCH_SIZE` (int, default=128): training batch size;
* `EPOCHS` (int, default=20): number of training epochs;
* `SCHEDULER` (str, default="warmuplinear"): learning rate scheduler (either "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", or "warmupcosinewithhardrestarts");
* `LR` (float, default=2e-5): learning rate;
* `WARMUP_RATIO` (float, default=0.1): ratio of warmup steps;
* `OPTIMIZER` (str, default="AdamW"): optimizer (either "AdamW" or "Adafacor");
* `WD` (float, default=0.01): weight decay;
* `FP16` (str, default=""): whether to use mixed precision training;
* `BF16` (str, default="--use_bf16"): whether to use bfloat16 precision training;
* `SEED` (int, default=42): random seed;
* `LOG_STEPS` (int, default=1): number of steps to log training metrics;
* `OUTPUT` (str, default="output/training"): output directory.

### Knowledge distillation

In addition to the standard fine-tuning approach, we provide a [knowledge distillation](https://www.sbert.net/examples/training/distillation/README.html) approach that can be performed by running:

```bash
bash scripts/run_single_vector_distillation.sh
```

after having set the following bash variables in the script:

* `DO_TRAIN="--do_train"`: setup training mode;
* `DO_TEST=""`: deactivate testing mode;
* `DATA="wikipedia"` or `DATA="minipile"`: training can be performed using the [Wikipedia](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/) or [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile) corpora;
* `TRAIN_LANG="en"`: training is performed in English.

Other relevant variables for training include:

* `TEACHER` (str, default="[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)"): checkpoint ID from Hugging Face for the teacher model;
* `STUDENT` (str, default="[facebook/xmod-base](https://huggingface.co/facebook/xmod-base)"): checkpoint ID from Hugging Face for the student model;
* `MAX_SEQ_LEN` (int, default=256): maximum input length for the text sequences (longer inputs are truncated);
* `POOL` (str, default="mean"): pooling strategy to distil a global sequence representation from the token-level contextualized embeddings (either "mean", "max", or "cls");
* `BATCH_SIZE` (int, default=128): training batch size;
* `EPOCHS` (int, default=20): number of training epochs;
* `OPTIMIZER` (str, default="AdamW"): optimizer (either "AdamW" or "Adafacor");
* `WD` (float, default=0.01): weight decay;
* `LR` (float, default=2e-5): learning rate;
* `SCHEDULER` (str, default="warmuplinear"): learning rate scheduler (either "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", or "warmupcosinewithhardrestarts");
* `WARMUP_RATIO` (float, default=0.1): ratio of warmup steps;
* `FP16` (str, default=""): whether to use mixed precision training;
* `BF16` (str, default="--use_bf16"): whether to use bfloat16 precision training;
* `SEED` (int, default=42): random seed;
* `LOG_STEPS` (int, default=1): number of steps to log training metrics;
* `SAVE_CKPTS_DURING_TRAINING` (str, default="--save_during_training"): whether to save checkpoints during training;
* `OUTPUT` (str, default="output/training"): output directory.

⚠️ Our preliminary results reveal that the distillation process is less effective than the standard training process. Fine-tuning DPR-XM on MS MARCO directly leads to a (small) dev performance of 32.7% MRR@10 in English and 23.5% MRR@10 in French (zero-shot), while distilling knowledge from BGE-v1.5 results in 23.9% MRR@10 for English and 12.2% MRR@10 for French (zero-shot).

### Evaluation

Evaluation can be performed using the `run_single_vector_biencoder.sh` script after setting the following bash variables:

* `DO_TEST="--do_test"`: setup testing mode;
* `DO_TRAIN=""`: deactivate training mode;
* `DATA="mmmarco"`: evaluation is performed on the [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) dataset;
* `LANG="xx"`: language code to evaluate on for mMARCO ("en", "fr", "ar", "de", "es", "hi", "id", "it", "ja", "nl", "pt", "ru", "vi", "zh").

## mono-XM: a modular cross-encoder

mono-XM is an [XMOD](https://arxiv.org/abs/2205.06266)-based cross-encoder model, as introduced in [monoBERT](https://arxiv.org/abs/1910.14424).

### Training

Training can be performed by running:

```bash
bash scripts/run_cross_encoder.sh
```

after having set the following bash variables in the script:

* `DO_TRAIN="--do_train"`: setup training mode;
* `DO_TEST=""`: deactivate testing mode;
* `DATA="mmarco"`: training uses the [msmarco-hard-negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives) dataset;
* `LANG="en"`: training is performed on the English samples of mMARCO;

Other relevant variables for training include:

* `MODEL` (str, default="[facebook/xmod-base](https://huggingface.co/facebook/xmod-base)"): checkpoint ID from Hugging Face;
* `MAX_SEQ_LEN` (int, default=512): maximum input length for the concatenated query+document (longer inputs are truncated);
* `BATCH_SIZE` (int, default=32): training batch size;
* `EPOCHS` (int, default=10): number of training epochs;
* `OPTIMIZER` (str, default="AdamW"): optimizer (either "AdamW" or "Adafacor");
* `LR` (float, default=2e-5): learning rate;
* `SCHEDULER` (str, default="warmuplinear"): learning rate scheduler (either "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", or "warmupcosinewithhardrestarts");
* `WD` (float, default=0.01): weight decay;
* `FP16` (str, default=""): whether to use mixed precision training;
* `BF16` (str, default="--use_bf16"): whether to use bfloat16 precision training;
* `SEED` (int, default=42): random seed;
* `LOG_STEPS` (int, default=1): number of steps to log training metrics;
* `OUTPUT` (str, default="output/training"): output directory.

### Evaluation

Evaluation can be performed using the same script after setting the following bash variables:

* `DO_TEST="--do_test"`: setup testing mode;
* `DO_TRAIN=""`: deactivate training mode;
* `DATA="mmmarco"`: evaluation is performed on the [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) dataset;
* `LANG="xx"`: language code to evaluate on for mMARCO ("en", "fr", "ar", "de", "es", "hi", "id", "it", "ja", "nl", "pt", "ru", "vi", "zh").
