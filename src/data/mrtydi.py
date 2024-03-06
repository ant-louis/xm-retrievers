import sys
import csv
import pathlib
from itertools import chain
from os.path import exists, join
from typing import List, Optional

import ir_datasets
from datasets import load_dataset

try:
    from src.utils.common import MRTYDI_LANGUAGES
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import MRTYDI_LANGUAGES


class MrTydiColbertLoader:
    def __init__(
        self, 
        lang: str,  # Language in which Mr. Tydi is loaded.
        load_test: Optional[str] = True, # Whether to load the test data.
        load_train: Optional[str] = False, # Whether to load the training data.
        data_folder: Optional[str] = 'data/mrtydi',  # Folder in which to save the downloaded datasets.
    ):
        assert lang in MRTYDI_LANGUAGES.keys(), f"Language {lang} not supported."
        assert load_train == False, "Loading training queries from Mr. Tydi is not implemented yet."
        self.lang = lang
        self.load_test = load_test
        self.load_train = load_train
        self.data_folder = data_folder

    def run(self):
        docid_mapping = {}
        data_filepaths = {}
        language = MRTYDI_LANGUAGES.get(self.lang)[0]

        # Load collection of passages.
        save_path = join(self.data_folder, f"{language}_collection.tsv")
        if not exists(save_path):
            dataset = load_dataset('castorini/mr-tydi-corpus', language, 'train')
            with open(save_path, 'w', newline='') as fOut:
                writer = csv.writer(fOut, delimiter='\t')
                for idx, sample in enumerate(dataset):
                    docid_mapping[sample['docid']] = idx
                    writer.writerow([idx, sample['title'] + ' ' + sample['text'].replace('\n', ' ').replace('\r', ' ')])
        data_filepaths['collection'] = save_path

        if self.load_test:
            # Load test queries.
            save_path = join(self.data_folder, f"{language}_queries.test.tsv")
            if not exists(save_path):
                dataset = ir_datasets.load(f"mr-tydi/{self.lang}/test")
                with open(save_path, 'w', newline='') as fOut:
                    writer = csv.writer(fOut, delimiter='\t')
                    for sample in dataset.queries_iter():
                        writer.writerow([sample.query_id, sample.text.replace('\n', ' ').replace('\r', ' ')])
            data_filepaths['test_queries'] = save_path

            # Load test qrels.
            save_path = join(self.data_folder, f"{language}_qrels.test.tsv")
            if not exists(save_path):
                dataset = ir_datasets.load(f"mr-tydi/{self.lang}/test")
                with open(save_path, 'w', newline='') as fOut:
                    writer = csv.writer(fOut, delimiter='\t')
                    for sample in dataset.qrels_iter():
                        writer.writerow([sample.query_id, 0, docid_mapping.get(sample.doc_id), 1])
            data_filepaths['test_qrels'] = save_path
        
        return data_filepaths
