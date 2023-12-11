#%% libraries
import os
from dataclasses import dataclass, field
from datasets import load_dataset
import torch
from typing import List, Optional
from collections import Counter
import json

from utils import make_dir, unlist, mode, save_json
# from universal_classes import Mention, Entity, Relation, Example, Dataset

#%% classes
@dataclass
class DocREDWord:
    annotation: str
    in_sentence_index: int
    parent_sentence: ...
    

    def _get_index(self):
        previous_sentences = self.parent_sentence.parent_example.sentences
        self.index = sum([len(el) for el in previous_sentences]) + self.in_sentence_index

    def __post_init__(self):
        self._get_index()
        
@dataclass
class DocREDSentence:
    annotation: List[str]
    index: int
    parent_example: ...

    def _get_words(self):
        self.words = [DocREDWord(el, i, self) for i, el in enumerate(self.annotation)]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

    def __post_init__(self):
        self._get_words()

@dataclass
class DocREDMention:

    annotation: ...
    
    parent_entity: ...

    def _get_string(self):
        self.string = self.annotation['name']

    def _get_in_sentence_word_span(self):
        self.in_sentence_word_span = (self.annotation['pos'][0], self.annotation['pos'][1] - 1)

    def _get_sentence_index(self):
        self.sentence_index = self.annotation['sent_id']

    def _get_type(self):
        self.type = self.annotation['type']

    def _get_words(self):
        sentence = self.parent_entity.parent_example.sentences[self.sentence_index]
        self.words = [sentence[i] for i in range(*self.annotation['pos'])]

    def _get_span(self):
        self.span = (self.words[0].index, self.words[-1].index)

    def __post_init__(self):
        self._get_string()
        self._get_sentence_index()
        self._get_in_sentence_word_span()
        self._get_type()
        self._get_words()
        self._get_span()

@dataclass
class DocREDEntity:

    annotation: ...

    parent_example: ...

    def _get_mentions(self):
        self.mentions = list(DocREDMention(el, self) for el in self.annotation)

    def _get_type(self):
        self.type = mode([el.type for el in self.mentions])

    def __hash__(self):
        
        info = [el.span for el in self.mentions]
        info.append(self.type)

        return hash(tuple(info))

    def __post_init__(self):
        self._get_mentions()
        self._get_type()

@dataclass
class DocREDRelation:
    
    head_index: int
    tail_index: int
    type: str

    parent_example: ...
    
    def _get_entities(self):
        self.head = self.parent_example.entities[self.head_index]
        self.tail = self.parent_example.entities[self.tail_index]
    

    def __post_init__(self):
        self._get_entities()

@dataclass
class DocREDExample:

    annotation: ...

    '''
    tokens
    
    candidate_spans
    candidate_span_pairs
    candidate_cluster_pairs
    
    full_clusters     
    cluster_relations    
    
    span_labels
    cluster_labels
    relation_labels
    '''

    def _get_sentences(self):
        raw_sentences = self.annotation['sents']
        self.sentences = list()
        for i, el in enumerate(raw_sentences):
            self.sentences.append(DocREDSentence(el, i, self))

    def _get_words(self):
        self.words = unlist([el.words for el in self.sentences])           
    
    def _get_entities(self):
        self.entities = [DocREDEntity(el, self) for el in self.annotation['vertexSet']]

    def _get_relations(self):
        labels = self.annotation['labels']
        triples = list(zip(labels['head'], labels['tail'], labels['relation_text']))
        
        self.relations = [DocREDRelation(el[0], el[1], el[2], self) for el in triples]

    def __post_init__(self):
        self._get_sentences()
        self._get_words()
        self._get_entities()
        self._get_relations()

@dataclass
class DocREDDataset:
    
    huggingface_dataset: ...
    relation_type_converter: Optional[dict] = field(default = None)
    counts_save_filename: Optional[dict] = field(default = 'data/processed/DocRED/relation_type_counts.json')
                
    def _get_examples(self):
        self.examples = [DocREDExample(el) for el in self.huggingface_dataset]

    def _get_relation_type_converter(self):
        relations = unlist([el.relations for el in self.examples])
        types = [el.type for el in relations]
        counts = Counter(types)
        save_json(counts, self.counts_save_filename)
        
        counts = counts.most_common()
        relation_types_by_frequency = list(zip(*counts))[0]

        self.relation_type_converter = dict(zip(relation_types_by_frequency, range(len(relation_types_by_frequency))))
    
    def _convert_relation_types(self):
        relations = unlist([el.relations for el in self.examples])
        for el in relations:
            el.type_index = self.relation_type_converter[el.type]

    def analyze_relations(self):
        relations = unlist([el.relations for el in self.examples])
        return [(el.head.type, el.tail.type, el.type, el.type_index) for el in relations]

    def __post_init__(self):
        self._get_examples()
        if not self.relation_type_converter:
            self._get_relation_type_converter()
        self._convert_relation_types()

#%% processing
output_data_path = 'data/processed/DocRED'
make_dir(output_data_path)

huggingface_dataset = load_dataset('docred')

train_data = huggingface_dataset['train_annotated']
valid_data = huggingface_dataset['validation']
test_data = huggingface_dataset['test']

print('train')
train_data = DocREDDataset(train_data)
print('valid')
valid_data = DocREDDataset(valid_data, train_data.relation_type_converter)
print('test')
test_data = DocREDDataset(test_data, train_data.relation_type_converter)

#%%
relation_combinations_train = train_data.analyze_relations()
relation_combinations_valid = valid_data.analyze_relations()
relation_combinations_test = test_data.analyze_relations()
torch.save(relation_combinations_train, os.path.join(output_data_path, 'relation_combinations_train.save'))
torch.save(relation_combinations_valid, os.path.join(output_data_path, 'relation_combinations_valid.save'))
torch.save(relation_combinations_test, os.path.join(output_data_path, 'relation_combinations_test.save'))

#%%
train_data = train_data.return_Dataset()
valid_data = valid_data.return_Dataset()
test_data = test_data.return_Dataset()

torch.save(train_data, os.path.join(output_data_path,'train_data.save'))    
torch.save(valid_data, os.path.join(output_data_path,'valid_data.save'))    
torch.save(test_data, os.path.join(output_data_path,'test_data.save'))   

