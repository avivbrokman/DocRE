#%% libraries
from os import path
from datasets import load_dataset, concatenate_datasets
import torch
import spacy
# import scispacy

from transformers import AutoTokenizer

from utils import make_dir
from spacy_data_classes import Dataset, tokenize_on_special_characters




#%% processing
# tokenizer
checkpoint = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# make data paths and directories
output_data_path = path.join('data', 'processed', 'BioRED', checkpoint)

make_dir(output_data_path)

# spacy model
text_processor = spacy.load("en_core_sci_lg")
text_processor = tokenize_on_special_characters(text_processor, ['-', '/'])

# load dataset from hugginface
huggingface_dataset = load_dataset('bigbio/biored')


# make all splits of dataset
train_data = huggingface_dataset['train']
valid_data = huggingface_dataset['validation']
test_data = huggingface_dataset['test']

train_valid_data = concatenate_datasets([train_data, valid_data])
full_data = concatenate_datasets([train_data, valid_data, test_data])

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

# for el_data in [train_data, valid_data, test_data, train_valid_data, full_data]:
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
# full_data = Dataset(full_data, text_processor, tokenizer)
# entity_types = full_data.get_entity_types()
# relation_types = full_data.get_relation_types()


# processes all datasets
print('\n train dataset \n')
# train_data = Dataset(train_data, text_processor, tokenizer, entity_types, relation_types)
# train_data = Dataset(train_data, text_processor, tokenizer)
print('\n validation dataset \n')
# valid_data = Dataset(valid_data, text_processor, tokenizer, entity_types, relation_types)
# valid_data = Dataset(valid_data, text_processor, tokenizer)
print('\n test dataset \n')
# test_data = Dataset(test_data, text_processor, tokenizer, entity_types, relation_types)
test_data = Dataset(test_data, text_processor, tokenizer)
# print('parsing train + validation dataset')
# train_valid_data = BioREDDataset(train_valid_data)

# filtering poorly annotated examples
# train_data.filter_by_title('The Secret of the Nagas')
# valid_data.filter_by_title('Paul Morphy')


#%% More processing
# relation_combinations_train = train_data.analyze_relations()
# relation_combinations_valid = valid_data.analyze_relations()
# relation_combinations_test = test_data.analyze_relations()
# torch.save(relation_combinations_train, os.path.join(output_data_path, 'relation_combinations_train.save'))
# torch.save(relation_combinations_valid, os.path.join(output_data_path, 'relation_combinations_valid.save'))
# torch.save(relation_combinations_test, os.path.join(output_data_path, 'relation_combinations_test.save'))

#%% More processing

entity_class_converter = train_data.entity_class_converter
relation_class_converter = train_data.relation_class_converter

# Saves everything
torch.save(entity_types, path.join(output_data_path,'entity_types.save'))
torch.save(relation_types, path.join(output_data_path,'relation_types.save'))
torch.save(entity_class_converter, path.join(output_data_path,'entity_class_converter.save'))
torch.save(relation_class_converter, path.join(output_data_path,'relation_class_converter.save'))

torch.save(train_data, path.join(output_data_path,'train_data.save'))    
torch.save(valid_data, path.join(output_data_path,'validation_data.save'))    
torch.save(test_data, path.join(output_data_path,'test_data.save'))   
# torch.save(train_valid_data, path.join(output_data_path,'train_validation_data.save'))   

