#%% libraries
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from os import path

from spacy_data_classes import Dataset

#%% Data Module
class ELREDataModule(LightningDataModule):
    def __init__(self, dataset_name, lm_checkpoint, train_split = 'train', inference_split = 'validation'):
        super().__init__()
        self.dataset_name = dataset_name
        self.lm_checkpoint = lm_checkpoint
        self.train_split = train_split
        self.inference_split = inference_split

    def _get_dataset_dir(self, dataset_name, lm_checkpoint):
        return path.join(dataset_name, lm_checkpoint)

    def setup(self, stage = None):
        nlp = Dataset.load_nlp(self.dataset_name)

        self.train_data = Dataset.load(self.dataset_name, self.lm_checkpoint, self.train_split, nlp)
        self.validation_data = Dataset.load(self.dataset_name, self.lm_checkpoint, self.inference_split, nlp)
        self.test_data = Dataset.load(self.dataset_name, self.lm_checkpoint, 'test', nlp)

        try:
            self.entity_type_converter = self.train_data.entity_type_converter
            self.relation_type_converter = self.train_data.relation_type_converter
        except:
            try: 
                self.entity_type_converter = self.validation_data.entity_type_converter
                self.relation_type_converter = self.validation_data.relation_type_converter

            except:
                try: 
                    self.entity_type_converter = self.test_data.entity_type_converter
                    self.relation_type_converter = self.test_data.relation_type_converter
                except: 
                    pass

    def train_dataloader(self):
        return DataLoader(self.train_data, collate_fn = lambda x: x[0])

    def val_dataloader(self):
        return DataLoader(self.validation_data, collate_fn = lambda x: x[0])

    def test_dataloader(self):
        return DataLoader(self.test_data, collate_fn = lambda x: x[0])


