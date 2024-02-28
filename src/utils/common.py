import os, sys, pathlib
from time import perf_counter
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers.models import Transformer, Pooling
try:
    from src.utils.SentenceTransformer import SentenceTransformerCustom
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.SentenceTransformer import SentenceTransformerCustom


MMARCO_LANGUAGES = {
    'ar': ('arabic', 'ar_AR'),
    'de': ('german', 'de_DE'),
    'en': ('english', 'en_XX'),
    'es': ('spanish', 'es_XX'),
    'fr': ('french', 'fr_XX'),
    'hi': ('hindi', 'hi_IN'),
    'id': ('indonesian', 'id_ID'),
    'it': ('italian', 'it_IT'),
    'ja': ('japanese', 'ja_XX'),
    'nl': ('dutch', 'nl_XX'),
    'pt': ('portuguese', 'pt_XX'),
    'ru': ('russian', 'ru_RU'),
    'vi': ('vietnamese', 'vi_VN'),
    'zh': ('chinese', 'zh_CN'),
}
MRTYDI_LANGUAGES = {
    'ar': ('arabic', 'ar_AR'),
    'bn': ('bengali', 'bn_IN'),
    'en': ('english', 'en_XX'),
    'fi': ('finnish', 'fi_FI'),
    'id': ('indonesian', 'id_ID'),
    'ja': ('japanese', 'ja_XX'),
    'ko': ('korean', 'ko_KR'),
    'ru': ('russian', 'ru_RU'),
    'sw': ('swahili', 'sw_KE'),
    'te': ('telugu', 'te_IN'),
    'th': ('thai', 'th_TH'),
}
MIRACL_LANGUAGES = {
    'ar': ('arabic', 'ar_AR'),
    'bn': ('bengali', 'bn_IN'),
    'en': ('english', 'en_XX'),
    'es': ('spanish', 'es_XX'),
    'fa': ('persian', 'fa_IR'),
    'fi': ('finnish', 'fi_FI'),
    'fr': ('french', 'fr_XX'),
    'hi': ('hindi', 'hi_IN'),
    'id': ('indonesian', 'id_ID'),
    'ja': ('japanese', 'ja_XX'),
    'ko': ('korean', 'ko_KR'),
    'ru': ('russian', 'ru_RU'),
    'sw': ('swahili', 'sw_KE'),
    'te': ('telugu', 'te_IN'),
    'th': ('thai', 'th_TH'),
    'zh': ('chinese', 'zh_CN'),
}
ALL_LANGUAGES = {**MMARCO_LANGUAGES, **MRTYDI_LANGUAGES, **MIRACL_LANGUAGES}


def load_sbert_model(model_name: str, max_seq_length: int, pooling: str):
    embedding_model = Transformer(
        model_name_or_path=model_name,
        model_args={'token': os.getenv("HF")},
        max_seq_length=max_seq_length,
        tokenizer_args={'model_max_length': max_seq_length},
    )
    pooling_model = Pooling(word_embedding_dimension=embedding_model.get_word_embedding_dimension(), pooling_mode=pooling)
    return SentenceTransformerCustom(modules=[embedding_model, pooling_model])


def set_xmod_language(model, lang:str):
    """
    Set the default language code for the model. This is used when the language is not specified in the input.
    Source: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/xmod/modeling_xmod.py#L687
    """
    lang = lang.split('-')[0]
    if (value := ALL_LANGUAGES.get(lang)) is not None:
        model.set_default_language(value[1])
    else:
        raise KeyError(f"Language {lang} not supported.")


def prepare_xmod_for_finetuning(model):
    """
    Freeze the embeddings and language adapters of the model.
    Source: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/xmod/modeling_xmod.py#L700
    """
    if model.__class__.__name__ == 'XmodForSequenceClassification':
        model = model.roberta
    for parameter in model.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.layer:
        if layer.output.adapter_layer_norm is not None:
            for parameter in layer.output.adapter_layer_norm.parameters():
                parameter.requires_grad = False
        for parameter in layer.output.adapter_modules.parameters():
            parameter.requires_grad = False


class catchtime:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'


def log_step(funct):
    """ Decorator to log the time taken by a function to execute.
    """
    import timeit, datetime
    from functools import wraps
    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"- Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper


def read_json_file(path_or_url: str):
    """ Read a JSON file from a local path or URL.
    """
    import re
    import json
    import urllib.request
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if bool(url_pattern.match(path_or_url)):
        with urllib.request.urlopen(path_or_url) as f:
            return json.load(f)
    with open(path_or_url, 'r') as f:
        return json.load(f)


def set_seed(seed: int):
    """ Ensure that all operations are deterministic on CPU and GPU (if used) for reproducibility.
    """
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model, verbose=True):
    """ Count the number of trainable parameters in a model.
    """
    all_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    if verbose:
        print(f"Trainable params: {round(trainable_params/1e6, 1)}M || All params: {round(all_params/1e6, 1)}M || Trainable ratio: {round(100 * trainable_params / all_params, 2)}%")
    return trainable_params


def push_to_hub(ressource_type: str, ressource_path: str, username: str, repo_id: str, create_repo: bool, private_repo: bool = True):
    import os
    from huggingface_hub import HfApi
    assert ressource_type in ["dataset", "model"], "ressource_type must be either 'dataset' or 'model'"
    try:
        api = HfApi(token=os.getenv("HF"))
        if create_repo:
            api.create_repo(repo_id=repo_id, repo_type=ressource_type, private=private_repo)
        if os.path.isfile(ressource_path):
            api.upload_file(
                repo_id=f"{username}/{repo_id}",
                repo_type=ressource_type,
                path_or_fileobj=ressource_path,
                path_in_repo=os.path.basename(ressource_path),
            )
        elif os.path.isdir(ressource_path):
            api.upload_folder(
                repo_id=f"{username}/{repo_id}", 
                repo_type=ressource_type, 
                folder_path=ressource_path,
            )
    except Exception as e:
        print("An error occurred while uploading ressource to HuggingFace Hub:", str(e))
