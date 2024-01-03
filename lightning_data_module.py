#%% libraries
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from os import path

#%% Data Module
class ELREDataModule(LightningDataModule):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name

    def _get_dataset_path(self, dataset_name, split):
        return path.join('data', dataset_name, split, '.save')

    def setup(self, stage = None):
        self.train_data = torch.load(self._get_dataset_path(self.dataset_name, 'train'))
        self.validation_data = torch.load(self._get_dataset_path(self.dataset_name, 'validation'))
        self.test_data = torch.load(self._get_dataset_path(self.dataset_name, 'test'))

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


# %%
