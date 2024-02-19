#%% libraries
from os import path
from datasets import load_dataset, concatenate_datasets
import torch
import spacy
import re
from copy import deepcopy

from transformers import AutoTokenizer

from utils import make_dir
from spacy_data_classes import Dataset, TokenizerModification




#%% processing

dataset_name = 'BioRED'

# tokenizer
checkpoint = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# spacy model
nlp = spacy.load("en_core_sci_lg")

tokenize_dash = TokenizerModification(re.escape('-'), 'add', 'infix')
tokenize_backslash = TokenizerModification(re.escape('/'), 'add', 'infix')
#period_suffix = TokenizerModification(r'(?<=[^\s])\.', 'add', 'suffix')
#period_suffix = TokenizerModification(r'\.(?=\s)', 'add', 'suffix')
#period_suffix = TokenizerModification(r'\.', 'add', 'suffix')
#period_prefix = TokenizerModification(r'\.', 'add', 'prefix')
#period_infix = TokenizerModification(r'\.', 'add', 'infix')
# period_suffix = TokenizerModification(r'\.(?=\s|$)', 'add', 'suffix')
remove_bar = TokenizerModification(re.escape('|'), 'remove', 'infix')
tokenize_semicolon = TokenizerModification(re.escape(';'), 'add', 'infix')
tokenize_plus = TokenizerModification(re.escape('+'), 'add', 'infix')
tokenize_left_parenthesis = TokenizerModification(re.escape('('), 'add', 'infix')
tokenize_right_parenthesis = TokenizerModification(re.escape(')'), 'add', 'infix')
#tokenize_right_parenthesis_suffix = TokenizerModification(re.escape(')'), 'add', 'suffix')
#tokenize_right_parenthesis_prefix = TokenizerModification(re.escape(')'), 'add', 'prefix')
#tokenize_left_parenthesis_prefix = TokenizerModification(re.escape('('), 'add', 'prefix')
#tokenize_left_parenthesis_suffix = TokenizerModification(re.escape('('), 'add', 'suffix')
tokenize_percent = TokenizerModification(r'(%\w+)', 'add', 'infix')

tokenizer_modifications = [tokenize_dash, 
                           tokenize_backslash, 
                           tokenize_semicolon, 
                           tokenize_plus, 
                           remove_bar, 
                           tokenize_left_parenthesis, 
                           tokenize_right_parenthesis,
                          #  tokenize_right_parenthesis_suffix,
                          #  tokenize_left_parenthesis_prefix,
                          #  tokenize_right_parenthesis_prefix,
                          #  tokenize_left_parenthesis_suffix 
                           tokenize_percent
                           ]

nlp = TokenizerModification.modify(nlp, tokenizer_modifications)
# nlp = TokenizerModification.remove_final_period_special_rules(nlp)
nlp = TokenizerModification.remove_middle_period_rules(nlp)


# load dataset from hugginface
huggingface_dataset = load_dataset('bigbio/biored')


# make all splits of dataset
train_data = huggingface_dataset['train']
validation_data = huggingface_dataset['validation']
test_data = huggingface_dataset['test']

train_validation_data = concatenate_datasets([train_data, validation_data])
full_data = concatenate_datasets([train_data, validation_data, test_data])

#%% cleaning
# =============================================================================
# relation_types = set()
# for example in full_data:
#     for relation in example['relations']:
#         relation_types.add(relation['type'])
#         
# print(relation_types)
# 
# entity_types = set()
# for example in full_data:
#     for entity in example['entities']:
#         entity_types.add(entity['semantic_type_id'])
#         
# print(entity_types)
# =============================================================================

# for el_data in [train_data, validation_data, test_data, train_validation_data, full_data]:
#     for el_ex in el_data:
#         title = el_ex['passages'][0]['text'][0]
#         abstract = el_ex['passages'][1]['text'][0]
#         text = title + ' ' + abstract
#         for el_ent in el_ex['entities']:
#             desired_text = el_ent['text'][0]

#             char_start = el_ent['offsets'][0][0]
#             char_end = el_ent['offsets'][0][1]

#             empirical_text = text[char_start:char_end]

#             if desired_text != empirical_text:
#                 print('pmid: ', el_ex['pmid'])
#                 print('desired: ', desired_text)
#                 print('empirical: ', empirical_text)
#                 print('\n')


#%% data processing
# processes full dataset to get complete list of entity types and relation types
full_data = Dataset(full_data, nlp, tokenizer)
entity_types = full_data.get_entity_types()
relation_types = full_data.get_relation_types()

relation_types = ['Positive_Correlation', 'Negative_Correlation', 'Association']


# processes all datasets
print('\n train dataset \n')
train_data = Dataset(train_data, nlp, tokenizer, entity_types, relation_types)
print('\n validation dataset \n')
validation_data = Dataset(validation_data, nlp, tokenizer, entity_types, relation_types)
print('\n test dataset \n')
test_data = Dataset(test_data, nlp, tokenizer, entity_types, relation_types)
print('parsing train + validation dataset')
train_validation_data = Dataset(train_validation_data, nlp, tokenizer, entity_types, relation_types)

#%% More processing
# relation_combinations_train = train_data.analyze_relations()
# relation_combinations_valid = validation_data.analyze_relations()
# relation_combinations_test = test_data.analyze_relations()
# torch.save(relation_combinations_train, os.path.join(output_data_path, 'relation_combinations_train.save'))
# torch.save(relation_combinations_valid, os.path.join(output_data_path, 'relation_combinations_valid.save'))
# torch.save(relation_combinations_test, os.path.join(output_data_path, 'relation_combinations_test.save'))

#%% More processing

entity_class_converter = train_data.entity_class_converter
relation_class_converter = train_data.relation_class_converter

# Saves everything

train_data.save_class_types_and_converters(dataset_name)
Dataset.save_nlp(nlp, dataset_name)

train_data.save(dataset_name, checkpoint, 'train')
validation_data.save(dataset_name, checkpoint, 'validation')
test_data.save(dataset_name, checkpoint, 'test')
train_validation_data.save(dataset_name, checkpoint, 'train_validation')


