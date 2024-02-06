#%% To do list

#%% libraries
from dataclasses import dataclass
from itertools import combinations, product
from torch import Tensor
from collections import Counter, defaultdict
from Levenshtein import distance
import re
from os import path
import torch
import spacy
from spacy.util import compile_infix_regex, compile_suffix_regex, compile_prefix_regex
from spacy.vocab import Vocab

import general_spacy_data_classes
from general_spacy_data_classes import Token, Span, SpanGroup, Doc, TokenizerModification, SpanUtils, Relation, Example, EvalRelation, Dataset
from utils import unlist, mode, parentless_print, make_dir


#%% SpanUtils
        
@staticmethod
def type_fun(annotation):
    return annotation['type']

@staticmethod
def id_fun(annotation):
    ids_list = annotation['normalized']
    ids = [el['db_id'] for el in ids_list]
    ids = list(set(ids))

    return ids

SpanUtils.type = type_fun
SpanUtils.id = id_fun




#%% Example

def _get_pmid(self):
        title_passage = self.annotation['passages'][0]
        self.pmid = title_passage['document_id']

def _get_mentions(self):
    self.doc._.mentions = set()
    self.eval_mentions = list()
    
    annotated_mentions = self.annotation['passages'][0]['entities'] + self.annotation['passages'][1]['entities']
    
    for el in annotated_mentions:
        if not SpanUtils.is_discontinuous(el): #!!!!! add this into BioRED too?
            mention, eval_mention = self._get_mention(el)
            if mention:
                self.doc._.mentions.add(mention)
            self.eval_mentions.append(eval_mention)

def _get_train_relation(self, annotation):
    entity1 = self.doc.spans[annotation['arg1_id']]
    entity2 = self.doc.spans[annotation['arg2_id']]
    relation = Relation(entity1, entity2, 'CID')
    return relation

def _get_train_relations(self):
    self.doc._.relations = set()
    
    annotated_relations = self.annotation['passages'][0]['relations'] + self.annotation['passages'][1]['relations']

    for el in annotated_relations:
        try:
            relation = self._get_train_relation(el)
            self.doc._.relations.add(relation)
        except:
            pass

def _get_eval_relation(self, annotation):
    head_id = annotation['arg1_id']
    tail_id = annotation['arg2_id']
    head = next(el for el in self.eval_entities if el.id == head_id)
    tail = next(el for el in self.eval_entities if el.id == tail_id)
    type_ = 'CID'
    relation = EvalRelation(head, tail, type_)
    return relation


Example._get_pmid = _get_pmid
Example._get_mentions = _get_mentions
Example._get_train_relation = _get_train_relation
Example._get_train_relations = _get_train_relations
Example._get_eval_relation = _get_eval_relation

class CDRExample(Example):
    
    def _get_pmid(self):
            title_passage = self.annotation['passages'][0]
            self.pmid = title_passage['document_id']

    def _get_mentions(self):
        self.doc._.mentions = set()
        self.eval_mentions = list()
        
        annotated_mentions = self.annotation['passages'][0]['entities'] + self.annotation['passages'][1]['entities']
        
        for el in annotated_mentions:
            if not SpanUtils.is_discontinuous(el): #!!!!! add this into BioRED too?
                mention, eval_mention = self._get_mention(el)
                if mention:
                    self.doc._.mentions.add(mention)
                self.eval_mentions.append(eval_mention)

    def _get_train_relation(self, annotation):
        entity1 = self.doc.spans[annotation['arg1_id']]
        entity2 = self.doc.spans[annotation['arg2_id']]
        relation = Relation(entity1, entity2, 'CID')
        return relation

    def _get_train_relations(self):
        self.doc._.relations = set()
        
        annotated_relations = self.annotation['passages'][0]['relations'] + self.annotation['passages'][1]['relations']

        for el in annotated_relations:
            try:
                relation = self._get_train_relation(el)
                self.doc._.relations.add(relation)
            except:
                pass

    def _get_eval_relation(self, annotation):
        head_id = annotation['arg1_id']
        tail_id = annotation['arg2_id']
        head = next(el for el in self.eval_entities if el.id == head_id)
        tail = next(el for el in self.eval_entities if el.id == tail_id)
        type_ = 'CID'
        relation = EvalRelation(head, tail, type_)
        return relation

