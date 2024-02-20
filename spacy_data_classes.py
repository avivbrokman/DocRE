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

from spacy.tokens import Token, Span, SpanGroup, Doc

from utils import unlist, mode, parentless_print, make_dir

# def generalized_lt(obj1, obj2, attr_names):
#     if self

#%% modify tokenizer
@dataclass
class TokenizerModification:
    regex: str
    action: str
    fix_type: str


    @staticmethod
    def modify(nlp, modifications):

        fix_dict = {'prefix': nlp.Defaults.prefixes,
                    'infix': nlp.Defaults.infixes,
                    'suffix': nlp.Defaults.suffixes
                    }
        for el in modifications:
            fix_list = fix_dict[el.fix_type]
            if el.action == 'add' and el.regex not in fix_list:
                fix_list.append(el.regex)
            elif el.action == 'remove' and el.regex in fix_list:
                fix_list.remove(el.regex)


        prefix_regex = compile_prefix_regex(fix_dict['prefix'])
        infix_regex = compile_infix_regex(fix_dict['infix'])
        suffix_regex = compile_suffix_regex(fix_dict['suffix'])

        nlp.tokenizer.prefix_search = prefix_regex.search
        nlp.tokenizer.infix_finditer = infix_regex.finditer
        nlp.tokenizer.suffix_search = suffix_regex.search

        return nlp

    @staticmethod
    def remove_final_period_special_rules(nlp):
        pattern = re.compile(r'^\w\.$')
        
        # Remove special cases that are single letters followed by a period
        keys_to_remove = [key for key in nlp.tokenizer.rules.keys() if pattern.match(key)]
        for key in keys_to_remove:
            del nlp.tokenizer.rules[key]

        return nlp

    @staticmethod
    def remove_middle_period_rules(nlp):
        # Preserve the original infix patterns
        original_infixes = nlp.Defaults.infixes
        
        # Exclude periods from infix patterns to avoid splitting at periods within words
        modified_infixes = tuple(infix for infix in original_infixes if infix != r'\.')
        
        # Add an exception for periods within words, e.g., 'p.Y67X'
        modified_infixes += (r'(?=[0-9a-zA-Z])\.(?=[0-9a-zA-Z])',)

        # Compile the modified infix patterns
        infix_regex = compile_infix_regex(modified_infixes)
        
        # Update the tokenizer with the new infix patterns
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        return nlp

 
#%%
# def tokenize_on_special_characters(nlp, special_characters):
#     # Get the default infix patterns from the tokenizer
#     infixes = nlp.Defaults.infixes

#     # Add custom infix patterns for the provided special characters
#     custom_infixes = [re.escape(char) for char in special_characters]
#     infixes = infixes + custom_infixes

#     # Compile new infix regex
#     infix_regex = compile_infix_regex(infixes)

#     # Create a new tokenizer with the custom infix pattern
#     nlp.tokenizer = Tokenizer(nlp.vocab, 
#                               prefix_search = nlp.tokenizer.prefix_search,
#                               suffix_search = nlp.tokenizer.suffix_search,
#                               infix_finditer = infix_regex.finditer,
#                               token_match = nlp.tokenizer.token_match)

#     return nlp

# def tokenize_on_final_periods(nlp):
#     # Get the default suffixes from the language defaults
#     suffixes = list(nlp.Defaults.suffixes)

#     # Add a pattern to match a period at the end of a word
#     # The pattern uses a positive lookbehind assertion to ensure the period is preceded by a non-whitespace character
#     period_suffix = r'(?<=[^\s])\.' r'\.(?=\s)'
#     if period_suffix not in suffixes:
#         suffixes.append(period_suffix)

#     # Compile the new suffix regex
#     suffix_regex = compile_suffix_regex(suffixes)

#     # Update the tokenizer with the new suffix search
#     nlp.tokenizer.suffix_search = suffix_regex.search

#     return nlp

# def remove_bar_tokenization(nlp):
#     # Get the default infix patterns and remove the one that splits at vertical bars
#     # current_infixes = nlp.Defaults.infixes
#     current_infixes = nlp.tokenizer.infix_finditer.pattern
#     infixes = [pattern for pattern in current_infixes if pattern != r'\|']
#     # Compile new infix regex
#     infix_regex = compile_infix_regex(infixes)

#     # Update the tokenizer with the new infix rule
#     nlp.tokenizer.infix_finditer = infix_regex.finditer

#     return nlp



#%% Modifying Spacy structures
Token.set_extension("subwords", default = None)
Token.set_extension("subword_indices", default = None)
Token.set_extension("sentence_index", default = None)

Span.set_extension("id", default = list())
# Span.set_extension("sentence_index", default = None)
Span.set_extension("subword_indices", default = None)

Doc.set_extension("mentions", default = set())
Doc.set_extension("relations", default = set())

#%% SpanUtils
class SpanUtils:
    
    @staticmethod
    def char_indices(annotation):
        start_char_index, end_char_index = annotation['offsets'][0]
        return start_char_index, end_char_index

    @staticmethod
    def text(annotation):
        return annotation['text'][0]    

    @staticmethod
    def type(annotation):
        return annotation['semantic_type_id']
    
    @staticmethod
    def id(annotation):
        id_string = annotation['concept_id']
        id_list = id_string.split(',')
        return id_list
    

    @staticmethod
    def is_discontinuous(annotation):
        return len(annotation['offsets']) > 1        

    @staticmethod
    def get_subword_indices(span):
        span._.subword_indices = (span[0]._.subword_indices[0], span[-1]._.subword_indices[1])
    
    @staticmethod
    def one_shift_subspans(span):
        subspans = set()
        left_subspan = span[1:]
        right_subspan = span[:-1]
        if left_subspan:
            SpanUtils.get_subword_indices(left_subspan)
            subspans.add(left_subspan)
        if right_subspan:
            SpanUtils.get_subword_indices(right_subspan)
            subspans.add(right_subspan)

        return subspans
    
    @staticmethod
    def subspans(span):
        subspans = set()
        # for start in range(span.start, span.end - 1):
        #     for end in range(start + 1, span.end):
        for start in range(len(span)):
            for end in range(start + 1, len(span) + 1):
                if  not(start == 0 and end == len(span)):
                    subspan = span[start:end]
                    SpanUtils.get_subword_indices(subspan)
                    subspans.add(subspan)

        return subspans
    
    @staticmethod
    def superspans(span, doc):
        superspans = set()
        if span.start != span.sent.start:
            left_superspan = doc[(span.start - 1):span.end]
            SpanUtils.get_subword_indices(left_superspan)
            superspans.add(left_superspan)
        if span.end != span.sent.end:
            right_superspan = doc[span.start:(span.end + 1)]
            SpanUtils.get_subword_indices(right_superspan)
            superspans.add(right_superspan)

        return superspans
    
    @staticmethod
    def has_noun(span):
        for token in span:
            if token.pos_ in ['NOUN', 'PROPN']:
                return True
        return False
    
    @staticmethod
    def has_no_subwords(span):
        return span._.subword_indices[0] == span._.subword_indices[1]

    @staticmethod
    def serialize_spans(obj, attr):
        return [(span.start_char, span.end_char, span.label_) for span in getattr(obj, attr)]
    
    @staticmethod
    def deserialize_spans(obj, attr, value):
        setattr(obj, attr, [obj.char_span(start, end, label = label) for start, end, label in value])

    @staticmethod
    def duplicate_spans(obj):
        if isinstance(obj, list):
            return [SpanUtils.duplicate_spans(el) for el in obj]
        if isinstance(obj, set):
            return {SpanUtils.duplicate_spans(el) for el in obj}
        elif isinstance(obj, Span):
            dup = obj.doc[obj.start:obj.end]
            dup._.subword_indices = obj._.subword_indices
            dup.label_ = obj.label_
            dup._.id = obj._.id
            return dup

        


#%% Saving utils
class SaveUtils:

    @staticmethod
    def Span2tuple(span):
        return (span.start_char, span.end_char, span.label_, span.id)

    @staticmethod
    def move_mentions_to_example(example):
        example.mentions = set(SaveUtils.Span2tuple(el) for el in example.doc._.mentions)
        example.doc._.mentions = None
    
    @staticmethod
    def tuple2Span(span_tuple, example):
        span = example.doc.char_span(span_tuple[0], span_tuple[1], label = span_tuple[2])
        span.id = span_tuple[4]
        return span
    
    @staticmethod
    def move_mentions_to_doc(example):
        example.doc._.mentions = set(SaveUtils.tuple2Span(el, example) for el in example.mentions)
    
    @staticmethod
    def remove_mentions_from_doc(example):
        example.doc._.mentions = None
    
    @staticmethod
    def span_group2mentions(example):
        span_groups = [el for el in example.doc.spans.values()]
        mentions = unlist(span_groups)  
        example.doc._.mentions = set(mentions)

    @staticmethod
    def Relation2tuple(relation):
        return (relation.head.attrs['id'], relation.tail.attrs['id'], relation.type)
    
    @staticmethod
    def move_relations_to_example(example):
        example.relations = [SaveUtils.Relation2tuple(el) for el in example.doc._.relations]
        example.doc._.relations = None

    @staticmethod
    def tuple2Relation(tuple, example):
        head_id = tuple[0]
        tail_id = tuple[1]
        type_ = tuple[2]

        head = example.doc.spans[head_id]
        tail = example.doc.spans[tail_id]
        relation = Relation(head, tail, type_)
        return relation
    
    @staticmethod
    def move_relations_to_doc(example):
        example.doc._.relations = [SaveUtils.tuple2Relation(el, example) for el in example.relations]
    
        



#%% EvalMention
@dataclass
class EvalMention:
    start_char: int
    end_char: int
    text: str
    type: str
    id: list[str]

    def __eq__(self, other):
        return (self.start_char, self.end_char, self.type) == (other.start_char, other.end_char, other.type)

    def __hash__(self):
        return hash((self.start_char, self.end_char, self.type))

    def __lt__(self, other):
        if self.start_char < other.start_char:
            return True
        elif self.start_char == other.start_char:
            if self.end_char < other.end_char:
                return True
            elif self.end_char == other.end_char:
                return self.type < other.type
        return False
    


            

    @classmethod
    def from_annotation(cls, annotation):
        start_char, end_char = SpanUtils.char_indices(annotation)
        text = SpanUtils.text(annotation)
        type = SpanUtils.type(annotation)
        id_ = SpanUtils.id(annotation)

        return cls(start_char, end_char, text, type, id_)
    
    @classmethod
    def from_span(cls, span, type_ = None):
        start_char, end_char = span.start_char, span.end_char
        text = span.text
        id_ = span._.id

        if type_ is None:
            type_ = span.label_

        return cls(start_char, end_char, text, type_, id_)
    
    def to_span(self, doc):
        span = doc.char_span(self.start_char, self.end_char)
        span.label_ = self.type
        span._.id = self.id
        SpanUtils.get_subword_indices(span)
        return span 


    # @classmethod
    # def from_prediction(cls, spacy_span, type = None):
    #     start_char, end_char = spacy_span._.subword_indices
    #     text = spacy_span.text
    #     id_ = spacy_span._.id
    #     return cls(start_char, end_char, text, type, id_)

#%% EvalSpanPair  
@dataclass
class EvalSpanPair:
    mention1: EvalMention
    mention2: EvalMention
    coref: int
    # type: str

    def __post_init__(self):
        self.mentions = {self.mention1, self.mention2}

    def __hash__(self):
        return hash((tuple(sorted(self.mentions)), self.coref))
    
    def __eq__(self, other):
        return self.coref == other.coref and self.mentions == other.mentions
        
    @classmethod
    def from_span_pair(cls, span_pair, coref = None):
        mention1 = EvalMention.from_span(span_pair.span1)
        mention2 = EvalMention.from_span(span_pair.span2)

        # type_ = span_pair.span1.label_
        if coref is None:
            coref = span_pair.coref

        # return cls(mention1, mention2, coref, type_)
        return cls(mention1, mention2, coref)
    
#%% EvalEntity
@dataclass
class EvalEntity:
    mentions: set[EvalMention]
    type: str
    id: str

    def __eq__(self, other):
        return (self.mentions, self.type) == (other.mentions, other.type)

    def __hash__(self):
        return hash((tuple(sorted(self.mentions)), self.type))    

    def __lt__(self, other):
        for self_el, other_el in zip(self.mentions, other.mentions):
            if self_el < other_el:
                return True
            if self_el > other_el:
                return False
        return self.type < other.type

    def __len__(self):
        return len(self.mentions)

    @classmethod
    def from_entity(cls, entity):

        mentions = {EvalMention.from_span(el) for el in entity}
        type_ = entity.attrs['type']
        id_ = entity.name

        return cls(mentions, type_, id_)
    
    # def does_overlap(self, other):
    #     self_offsets = {(el.start_char, el.end_char) for el in self.mentions}
    #     other_offsets = {(el.start_char, el.end_char) for el in other.mentions}

    #     return bool(self_offsets & other_offsets)
    
    def merge(self, other):
        self_offsets = {(el.start_char, el.end_char) for el in self.mentions}
        other_offsets = {(el.start_char, el.end_char) for el in other.mentions}

        if self_offsets & other_offsets:
        
            mentions = self.mentions | other.mentions
            type_ = mode([el.type for el in mentions])
            id_ = mode(unlist([el.id for el in mentions]))

            return EvalEntity(mentions, type_, id_)
        else:
            return False
        
    # @classmethod
    # def from_span(cls, span):
    #     mentions = {EvalMention.from_span(span)}
    #     type_ = span.label_
    #     id_ = span.id
    #     return cls(mentions, type_, id_)
    
    @classmethod
    def from_eval_mention(cls, mention):
        mentions = {mention}
        type_ = mention.type
        id_ = mention.id
        return cls(mentions, type_, id_)

    @classmethod
    def from_eval_span_pair(cls, span_pair):
        mentions = span_pair.mentions
        type_ = mode([el.type for el in mentions])
        id_ = mode(unlist([el.id for el in mentions]))
 
        return cls(mentions, type_, id_)
    
    #e2e inference
    def to_span_group(self, doc):
        spans = {el.to_span(doc) for el in self.mentions}
        span_group = SpanGroup(doc,
                               name = self.id[0] if self.id else '',
                               attrs = {'id': self.id, 'type': self.type},
                               spans = spans
                               )
        return span_group
    
    def eval_span_pairs(self):
        return set(EvalSpanPair(men1, men2, 1) for men1, men2 in combinations(self.mentions, 2)) 
    

#%% EvalRelation
@dataclass
class EvalRelation:
    head: EvalEntity
    tail: EvalEntity
    type: str

    def __eq__(self, other):
        return (self.entities, self.type) == (other.entities, other.type) 

    def __hash__(self):
        return hash((tuple(sorted(self.entities)), self.type))  

    def __post_init__(self):
        self.entities = {self.head, self.tail}

    @classmethod
    def from_relation(cls, relation, type_):
        head = EvalEntity.from_entity(relation.head)
        tail = EvalEntity.from_entity(relation.tail)
        
        return cls(head, tail, type_)

#%% SpanPair
@dataclass
class SpanPair:
    span1: Span
    span2: Span
    coref: int = None
    
    def __hash__(self):
        # return hash((self.span1, self.span2, self.coref))
        return hash((tuple(sorted(self.spans)), self.coref))

    def __eq__(self, other):
        return self.spans == other.spans and self.coref == other.coref
    
    def __post_init__(self):
        self.spans = {self.span1, self.span2}
        self.type = self.span1.label_
        self.path = self.path_tokens()

    def levenshtein_distance(self):
        return distance(self.span1.text, self.span2.text)
    
    def intervening_token_indices(self):
        first_token = min(self.span1.start, self.span2.start)
        last_token = max(self.span1.end, self.span2.end)
        all_tokens = set(range(first_token, last_token))
        span1 = set(range(self.span1.start, self.span1.end))
        span2 = set(range(self.span2.start, self.span2.end))
        both_spans = span1 | span2
        intervening = all_tokens - both_spans
        return intervening
    
    def length_difference(self):
        return abs(len(self.span1.text) - len(self.span2.text))
    
    def token_distance(self):
        return len(self.intervening_token_indices())

    def sentence_distance(self):
        return abs(self.span1[0]._.sentence_index - self.span2[0]._.sentence_index)

    def intervening_span(self):
        intervening = self.intervening_token_indices()
        if intervening:
            intervening_span = (min(intervening), max(intervening) + 1)
            intervening_span = self.span1.doc[intervening_span[0]:intervening_span[1]]
            SpanUtils.get_subword_indices(intervening_span)
            return intervening_span
        
    def inclusive_span(self):
        first_token = min(self.span1.start, self.span2.start)
        last_token = max(self.span1.end, self.span2.end)
        intervening = set(range(first_token, last_token))
        if intervening:
            intervening_span = (min(intervening), max(intervening))
            intervening_span = self.span1.doc[intervening_span[0]:intervening_span[1]]
            SpanUtils.get_subword_indices(intervening_span)
            return intervening_span
        
    
    @staticmethod    
    def _get_lca(token1, token2):
        ancestors1 = set(token1.ancestors)
        for ancestor in token2.ancestors:
            if ancestor in ancestors1:
                return ancestor
        return None

    @staticmethod
    def _trace_path_to_LCA(token, lca):
        path = set([lca])
        current = token
        while current != lca:
            path.add(current)
            current = current.head
            
        return path
    
    @staticmethod
    def _connecting_path(token1, token2):

        path = set()
        lca = SpanPair._get_lca(token1, token2)
        if lca:
            path1 = SpanPair._trace_path_to_LCA(token1, lca)
            path2 = SpanPair._trace_path_to_LCA(token2, lca)
            path.update(path1)
            path.update(path2)
        return path

    def path_tokens(self):
            
        paths = [SpanPair._connecting_path(token1, token2) for token1, token2 in product(self.span1, self.span2)] 
            
        path_union = set.union(*paths)

        return path_union
    
    @staticmethod
    def filter_unequal_types(span_pairs):
        if isinstance(span_pairs, list):
            return [el for el in span_pairs if el.span1.label_ == el.span2.label_]
        if isinstance(span_pairs, set):
            return {el for el in span_pairs if el.span1.label_ == el.span2.label_}
    
#%% EntityUtils
class EntityUtils:
    
    @staticmethod
    def pos_span_pairs(entity):
       return [SpanPair(el1, el2, 1) for el1, el2 in combinations(entity, 2)]

    @staticmethod
    def neg_span_pairs_entity(entity, other):
        return [SpanPair(el1, el2, 0) for el1, el2 in product(entity, other) if el1 != el2]
    
    @classmethod
    def neg_span_pairs(cls, entity, others):
        # return unlist([cls.neg_span_pairs_entity(entity, el) for el in others if el.attrs['type'] == entity.attrs['type']])    
        return unlist([cls.neg_span_pairs_entity(entity, el) for el in others])    
    
    @staticmethod
    def from_span(span):
        spans = {span}
        type_ = span.label_
        id_ = span._.id
        span_group = SpanGroup(span.doc, 
                               name = id_,
                               attrs = {'id': id_, 'type': type_},
                               spans = spans
                               )
        return span_group

    @staticmethod
    def from_span_pair(span_pair):
        spans = span_pair.spans
        type_ = mode([el.type for el in spans])
        id_ = mode(unlist([el.id for el in spans]))
        doc = list(spans)[0].doc
        span_group = SpanGroup(doc, 
                               name = id_,
                               attrs = {'id': id_, 'type': type_},
                               spans = spans
                               )
        return span_group
    
    
    @staticmethod
    def merge(entity1, entity2):
        self_offsets = {(el.start_char, el.end_char) for el in entity1.mentions}
        other_offsets = {(el.start_char, el.end_char) for el in entity2.mentions}

        if self_offsets & other_offsets:
        
            mentions = entity1.spans | entity2.spans
            type_ = mode([el.type for el in mentions])
            id_ = mode(unlist([el.id for el in mentions]))

            span_group = SpanGroup(entity1.doc, 
                                   name = id_,
                                   attrs = {'id': id_, 'type': type_},
                                   spans = mentions
                                   )

            return span_group
        else:
            return False
    
            
#%% Relation
@dataclass
class Relation:
    head: SpanGroup
    tail: SpanGroup
    type: str = None
    
    def __eq__(self, other):
        return {self.head, self.tail} == {other.head, other.tail} and self.type == other.type
    
    def __hash__(self):
        hash_list = list({self.head, self.tail})
        hash_list.append(self.type)
        hash_tuple = tuple(hash_list)
        return hash(hash_tuple)
    
    # @classmethod
    # def from_set(cls, entities, type_):
    #     head = entities.pop()
    #     tail = entities.pop()

    #     return Relation(head, tail, type_)
    
    def _null_relation(self):
        return Relation(self.head, self.tail, None)
    
    def is_valid_relation(self, valid_triples):
        combination = {self.head.type, self.tail.type, self.type}
        return combination in valid_triples
    
    def is_valid_entity_pair(self, valid_doubles):
        combination = {self.head.type, self.tail.type}
        return combination in valid_doubles
    
    @staticmethod
    def filter_invalid_relations(relations, valid_triples):
        return set(el for el in relations if el.is_valid_relation(valid_triples))
    
    @staticmethod
    def filter_invalid_entity_pairs(relations, valid_doubles):
        return set(el for el in relations if el.is_valid_entity_pair(valid_doubles))
    
    @staticmethod
    def filter_relations(entity_pairs, relations):
       return set(entity_pairs) - set(relations)

    def enumerate_span_pairs(self):
        return [SpanPair(el1, el2) for el1, el2 in product(self.head, self.tail)]   

    @staticmethod
    def candidate_relations(entities):
        return [Relation(el1, el2) for el1, el2 in combinations(entities.values(), 2)] 

    # def _mutate_relation_type(self, relation_type):
    #     return Relation(self.head, self.tail, relation_type)
    
    # def relation_type_mutations(self, relation_types, positive_relations):
    #     #relation_types = relation_class_converter.class2index_dict.keys()
    #     mutated_relations = set(self._mutate_relation_type(el) for el in relation_types)
    #     negative_relations = self._filter_positive_relations(mutated_relations, positive_relations)
    #     return negative_relations

    # def _mutate_entity(self, entity, head_or_tail):
    #     if head_or_tail == 'head':
    #         return Relation(entity, self.tail, self.type)
    #     elif head_or_tail == 'tail':
    #         return Relation(self.head, entity, self.type)

    # def entity_mutations(self, entities, positive_relations):
    #     entities = set(entities) - {self.head, self.tail}
    #     head_mutations = set(self._mutate_entity(el, 'head') for el in entities if el.attrs['type'] == self.head.attrs['type'])
    #     tail_mutations = set(self._mutate_entity(el, 'tail') for el in entities if el.attrs['type'] == self.tail.attrs['type'])

    #     mutations = head_mutations | tail_mutations

    #     negative_relations = self._filter_positive_relations(mutations, positive_relations)
        
    #     return negative_relations

    # def negative_relations(self, relation_types, positive_relations, entities):
    #     negative_pairs = set()
    #     negative_pairs.update(self.relation_type_mutations(relation_types, positive_relations))
    #     negative_pairs.update(self.entity_mutations(entities, positive_relations))

    #     return negative_pairs

    
        

#%% Example
@parentless_print
@dataclass
class Example:

    annotation: ...
    parent_dataset: ...
    
    '''
    tokens
    
    candidate_spansscorer
    candidate_span_pairs
    candidate_cluster_pairs
    
    full_clusters     
    cluster_relations    
    
    span_labels
    cluster_labels
    relation_labels
    '''

    def __post_init__(self):
        self._get_pmid()
        self._get_text()
        self._get_doc()
        self._get_sentence_indices()
        self._tokenize()

        self._get_mentions()
        self._get_entities()
        self._get_relations()

    def _get_pmid(self):
        self.pmid = self.annotation['pmid']

    def _get_text(self):
        passages = self.annotation['passages']
        for el in passages:
            if el['type'] == 'title':
                self.title = el['text'][0]
            if el['type'] == 'abstract':
                self.abstract = el['text'][0]
        
        self.text = self.title + ' ' + self.abstract 
        
    def _get_doc(self):
        self.doc = self.parent_dataset.nlp(self.text)
    
    def _get_sentence_indices(self):
        for i, sentence in enumerate(self.doc.sents):
            for token in sentence:
                token._.sentence_index = i

    def _get_mention(self, annotation):
        start_char_index, end_char_index = SpanUtils.char_indices(annotation)
        type_ = SpanUtils.type(annotation)
        mention = self.doc.char_span(start_char_index, end_char_index, label = type_)
        if mention:
            mention._.id = SpanUtils.id(annotation)
            SpanUtils.get_subword_indices(mention)
        eval_mention = EvalMention.from_annotation(annotation)
        return mention, eval_mention
                
    def _get_mentions(self):
        self.doc._.mentions = set()
        self.eval_mentions = list()
        
        for el in self.annotation['entities']:
            if len(el['offsets']) == 1:
                mention, eval_mention = self._get_mention(el)
                if mention:
                    self.doc._.mentions.add(mention)
                self.eval_mentions.append(eval_mention)
    
    def _get_train_entities(self):
        entity_dict = defaultdict(set)
        for mention in self.doc._.mentions:
            for id_ in mention._.id: # this for loop exists b/c some mentions have multiple ids
                entity_dict[id_].add(mention)
        for id_, entity in entity_dict.items():
            type_ = list(entity)[0].label_
            self.doc.spans[id_] = SpanGroup(self.doc, 
                                 name = id_,
                                 attrs = {'id': id_, 'type': type_},
                                 spans = entity
                                 )
   
    def _get_eval_entities(self):
        self.eval_entities = list()
        eval_entity_dict = defaultdict(list)
        for mention in self.eval_mentions:
            for id_ in mention.id:
                eval_entity_dict[id_].append(mention)
        for id_, entity in eval_entity_dict.items():
            type = entity[0].type
            mentions = set(entity)
            self.eval_entities.append(EvalEntity(mentions, type, id_))

    def _get_eval_span_pairs(self):
        pairs_list = [el.eval_span_pairs() for el in self.eval_entities]
        self.eval_span_pairs = set.union(*pairs_list)

    def _get_entities(self):        
        self._get_train_entities()
        self._get_eval_entities()
        self._get_eval_span_pairs()
        
    def _get_train_relation(self, annotation):
        entity1 = self.doc.spans[annotation['concept_1']]
        entity2 = self.doc.spans[annotation['concept_2']]
        relation = Relation(entity1, entity2, annotation['type'])
        return relation
        
    def _get_train_relations(self):
        self.doc._.relations = set()
        for el in self.annotation['relations']:
            try:
                relation = self._get_train_relation(el)
                self.doc._.relations.add(relation)
            except:
                pass
    
    def _get_eval_relation(self, annotation):
        head_id = annotation['concept_1']
        tail_id = annotation['concept_2']
        head = next(el for el in self.eval_entities if el.id == head_id)
        tail = next(el for el in self.eval_entities if el.id == tail_id)
        type_ = annotation['type']
        relation = EvalRelation(head, tail, type_)
        return relation

    def _get_eval_relations(self):
        self.eval_relations = [self._get_eval_relation(el) for el in self.annotation['relations']]

    def _get_relations(self):
        self._get_train_relations()
        self._get_eval_relations()

    def _tokenize(self):
        for i, el in enumerate(self.doc):
            el._.subwords = self.parent_dataset.tokenizer.encode(el.text, add_special_tokens = False) #el.text_with_ws?
            if i == 0:
                el._.subword_indices = (0, len(el._.subwords))
            else:
                start_index = self.doc[i - 1]._.subword_indices[1]
                end_index = start_index + len(el._.subwords)
                el._.subword_indices = (start_index, end_index)

        # if self.pmid == '14722929':
        #     pass

        self.subword_tokens = unlist([el._.subwords for el in self.doc])
                
    # for NER training            
    def _subspans(self):
        
        one_shift_subspans = set()
        for el in self.doc._.mentions:
            one_shift_subspans.update(SpanUtils.one_shift_subspans(el))
              
        subspans = set()
        for el in self.doc._.mentions:
            subspans.update(SpanUtils.subspans(el))
        multi_shift_subspans = subspans - one_shift_subspans
        

        short_negative_spans = set()
        for start in range(len(self.doc) - 2):
            for end in (start + 1, start + 2):
                subspan = self.doc[start:end]
                SpanUtils.get_subword_indices(subspan)
                short_negative_spans.add(subspan)

        return one_shift_subspans, multi_shift_subspans, short_negative_spans
    
    def _superspans(self):
        superspans = set()
        for el in self.doc._.mentions:
            superspans.update(SpanUtils.superspans(el, self.doc))
        
        # for entity in self.doc.spans.values():
        #     superspan_list = [SpanUtils.superspans(span, self.doc) for span in entity]
        #     superspans.update(set.union(*superspan_list))           
                                
        return superspans
    
    def _filter_mentions(self, candidate_negative_spans):
        mentions = [(el.start, el.end) for el in self.doc._.mentions]
        return set(el for el in candidate_negative_spans if (el.start, el.end) not in mentions) 

    def negative_spans(self):
        
        one_shift_subspans, multi_shift_subspans, short_negative_spans = self._subspans()
        superspans = self._superspans()

        one_shift_subspans = self._filter_mentions(one_shift_subspans)
        multi_shift_subspans = self._filter_mentions(multi_shift_subspans)
        short_negative_spans = self._filter_mentions(short_negative_spans)
        superspans = self._filter_mentions(superspans)

        one_shift_spans = one_shift_subspans | superspans
        multi_shift_subspans -= one_shift_spans
        short_negative_spans -= one_shift_spans | multi_shift_subspans

        short_negative_spans = set(el for el in short_negative_spans if not SpanUtils.has_no_subwords(el))
        multi_shift_subspans = set(el for el in multi_shift_subspans if not SpanUtils.has_no_subwords(el))
        short_negative_spans = set(el for el in short_negative_spans if not SpanUtils.has_no_subwords(el))

        return one_shift_spans, multi_shift_subspans, short_negative_spans
    
    # for NER inference
    def candidate_spans(self):
        candidate_spans = set()
        candidate_spans.update(self.doc._.mentions)
        mention_indices = {(el.start, el.end) for el in self.doc._.mentions}
        for sentence in self.doc.sents:
            for start in range(sentence.start, sentence.end - 1):
                for end in range(start + 1, sentence.end):
                    span = self.doc[start:end]
                    SpanUtils.get_subword_indices(span)
                    if not SpanUtils.has_no_subwords(span) and (span.start, span.end) not in mention_indices:
                        candidate_spans.add(span)
        candidate_spans = SpanUtils.duplicate_spans(candidate_spans)
        return candidate_spans #e2e inference
    
    # For Coref training
    def positive_span_pairs(self):
        return unlist([EntityUtils.pos_span_pairs(el) for el in self.doc.spans.values()])
        
    def negative_span_pairs(self):
        entities = set(self.doc.spans.values())
        return unlist([EntityUtils.neg_span_pairs(el, entities - {el}) for el in entities])

    # For RC training
    # def negative_relations(self):
    #     return unlist([el.negative_relations(self.parent_dataset.relation_types, self.doc._.relations, self.doc.spans.values()) for el in self.doc._.relations])

    def candidate_relations(self):
       return Relation.candidate_relations(self.doc.spans)
    
    def filter_rare_relation_types(self):
        return [el for el in self.doc._.relations if el.type in self.parent_dataset.relation_types]

    def negative_relations(self):
        entity_pairs = self.candidate_relations()
        positive_relations = self.filter_rare_relation_types()
        negative_relations = Relation.filter_relations(entity_pairs, positive_relations)
        return list(negative_relations)
    

    
         

#%% ClassConverter
@dataclass
class ClassConverter:
    classes: list[str|None]

    def __post_init__(self):
        self.class2index_dict = dict(zip(self.classes, range(len(self.classes))))
        self.index2class_dict = dict(zip(range(len(self.classes)), self.classes))

    def class2index(self, object_):
        if isinstance(object_, list):
            return [self.class2index(el) for el in object_]
        elif isinstance(object_, set):
            return set(self.class2index(el) for el in object_)
        # elif object_ == '':
        #     return self.class2index_dict[None]
        elif isinstance(object_, str) or object_ is None:
            return self.class2index_dict[object_]
    
    def index2class(self, object_):
        if isinstance(object_, list):
            return [self.index2class(el) for el in object_]
        elif isinstance(object_, set):
            return set(self.index2class(el) for el in object_)
        elif isinstance(object_, Tensor):
            return self.index2class_dict[object_.item()]
        elif isinstance(object_, int):
            return self.index2class_dict[object_]
        
    def __len__(self):
        return len(self.classes)
    
#%% Dataset
@dataclass
class Dataset:
    
    huggingface_dataset: ...
    nlp: ...
    tokenizer: ...

    entity_types: list[str] = None
    relation_types: list[str] = None

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):

        return self.examples[idx]

    def __post_init__(self):
        self.process()
        self._get_examples()

    def _get_examples(self):
        self.examples = [Example(el, self) for el in self.huggingface_dataset]

    def _get_type_converter(self, types, null_value = None):
        types = types + [null_value]
        return ClassConverter(types)

    def get_entity_types(self):
        counts = Counter()
        for example in self.examples:
            for entity in example.eval_entities:
                counts.update([entity.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]

        return list(types)
        
    def get_relation_types(self):
        counts = Counter()
        for example in self.examples:
            for relation in example.eval_relations:
                counts.update([relation.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        print(counts)

        return types

    def process(self):
        # (a) entity types don't exist
        # (b) entity types
        if self.entity_types:
            self.entity_class_converter = self._get_type_converter(self.entity_types, '')
        if self.relation_types:
            self.relation_class_converter = self._get_type_converter(self.relation_types, None)

   
    def save_class_types_and_converters(self, dataset_name):
        save_dir = path.join('data', 'processed', dataset_name)
        make_dir(save_dir)

        torch.save(self.entity_types, path.join(save_dir, 'entity_types.save'))  
        torch.save(self.relation_types, path.join(save_dir, 'relation_types.save'))  

        torch.save(self.entity_class_converter, path.join(save_dir, 'entity_type_converter.save'))
        torch.save(self.relation_class_converter, path.join(save_dir, 'relation_type_converter.save'))

    @staticmethod
    def save_nlp(nlp, dataset_name):
        save_dir = path.join('data', 'processed', dataset_name)
        filepath = path.join(save_dir, 'nlp.spacy_model')
        make_dir(save_dir)
        nlp.to_disk(filepath)

    @staticmethod
    def load_nlp(dataset_name):
        nlp = spacy.load('en_core_sci_lg')
        
        filepath = path.join('data', 'processed', dataset_name, 'nlp.spacy_model')

        nlp.from_disk(filepath)

        return nlp

    def save(self, dataset_name, lm_checkpoint, split):
        # directories
        data_path = path.join('data', 'processed', dataset_name, lm_checkpoint)
        split_path = path.join(data_path, split)
        
        make_dir(split_path)

        # detach mentions and relations from spacy do
        for example in self.examples:
            SaveUtils.remove_mentions_from_doc(example)
            SaveUtils.move_relations_to_example(example)
        # Save spaCy docs
        for i, example in enumerate(self.examples):
            example.doc.to_disk(path.join(split_path, f'doc_{i}.spacy'))

        # Save dataset without spaCy docs
        for example in self.examples:
            example.doc = None
        torch.save(self, path.join(split_path, 'data.save'))

    @staticmethod
    def load(dataset_name, lm_checkpoint, split, nlp):
        
        data_path = path.join('data', 'processed', dataset_name, lm_checkpoint)
        split_path = path.join(data_path, split)

        # Load dataset
        dataset = torch.load(path.join(split_path, 'data.save'))

        # Reload spaCy docs
        for i, example in enumerate(dataset.examples):
            example.doc = Doc(nlp.vocab).from_disk(path.join(split_path, f'doc_{i}.spacy'))
            example._get_sentence_indices()
            example._tokenize()

            SaveUtils.span_group2mentions(example)
            SaveUtils.move_relations_to_doc(example)

        return dataset