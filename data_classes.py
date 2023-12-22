# doing the traditional stuff

# get entity-type-pair/relation_type valid combinations
    # help in choosing HARD examples for contrastive learning (this is only for later)
    # post-processing so we remove obvious FP's    
# get negative examples
    # choose what proportion are negative in each combination
# get this all into a tensor so we can predict on it.  
# get all of the extra info for contextual embeddings, etc.

#%% libraries
import os
import torch
from dataclasses import dataclass, field
from Levenshtein import distance
from typing import Optional
from itertools import combinations, product

from utils import make_dir, unlist

#%% body
# Token gets tokenized index
# recursive function that gets tokens no matter how deeply nested? or Span, Sentence, Example have those attributes or as a function?
##### Span stuff
# Span needs __len__ method
### training
# Example needs negative spans.  Need to figure out how to generate hard negative spans.  One way is to first only include spans with nouns.  Another way is to only include subspans and superspans as negative examples.  That's probably the main way. And only within sentence.
### eval
# Example needs all possible intra-sentence spans enumerated
# functions that evaluate in typed/untyped mode
##### Clustering
# Span needs levenshtein distance method between two spans
### train
# Example needs negative pairs of spans.  Could do all pairs that don't coref.  But want harder ones.  How to do that?  Hard negatives could be only same type.  Hard negatives could have closer levenshtein distances.
### eval
# function that creates all possible pairs of mentions
# functions/classes that evaluate in a strict mode/relaxed mode, span/string mode, typed/untyped mode
##### Relation classification
# Span needs token distance and sentence distance from other Span methods
# Span needs method to obtain all tokens between it and another span
### train
# Example needs hard negatives.  Perhaps for every possible example, switching out relation types, head entities, or tail entities.  But then need to check none of those are positives as well.
### eval
# function that creates all possible pairs of entities
# functions/classes that evaluate in a strict mode/relaxed mode, span/string mode, typed, untyped mode

### Loss

##### overall
# Token
# Span
# Sentence
# Mention (subclass of Span)
# Cluster
# Entity? (subclass of Cluster)
# EntityPair?/Relation?
# Example
# Dataset
#%% Token
@dataclass
class Token:
    string: str
    in_sentence_index: int
    index: int
    # tokenizer_indices: int
    parent_sentence: Sentence

    def _get_subword_tokens(self):
        tokenizer = self.parent_sentence.parent_example.parent_dataset.tokenizer
        self.subword_tokens = tokenizer.encode(self.string, add_special_tokens = False)

    def _get_subword_indices(self):
        # start = sum(len(el.subword_tokens) for el in self.parent_sentence.tokens[:index]) - 1
        if self.index == 0
            start = 0
        else:
            start = self.parent_sentence[index - 1]
        end = start + num_subwords
        self.subword_indices = (start, end)


    def __post_init__(self):
        self._get_subword_tokens()
        self._get_subword_indices()
#%% Span
@dataclass(frozen = True)
class Span:
    tokens: list[Token]
    type: Optional[str]

    def _get_in_sentence_indices(self):
        self.in_sentence_indices = (tokens[0].in_sentence_index, tokens[-1].in_sentence_index + 1)

    def _get_indices(self):
        self.indices = (tokens[0].index, tokens[-1].index + 1)    

    def _get_parent_sentence(self):
        self.parent_sentence = self.tokens[0].parent_sentence

    def _get_string(self):
        self.string = [el.string for el in self.tokens].join(' ') 

    def _get_subword_indices(self):
        self.subword_indices = (self.tokens[0].subword_indices[0], self.tokens[-1].subword_indices[1])

    def __len__(self):
        return len(range(*self.indices))

    def __hash__(self):
        return hash(indices)

    def levenshtein_distance(self, other):
        return distance(self.string, other.string)

    def token_distance(self, other):
        first_token = min(self.indices[0], other.indices[0])
        last_token = min(self.indices[1], other.indices[1])
        all_tokens = set(range(first_token, last_token))
        span1 = set(*self.indices)
        span2 = set(*other.indices)
        overlap = span1 & span2
        return len(all_tokens - overlap)

    def sentence_distance(self, other):
        return abs(other.parent_sentence.index - self.parent_sentence.index)

    def intervening_span(self, other):
        first_token = min(self.indices[0], other.indices[0])
        last_token = min(self.indices[1], other.indices[1])
        all_tokens = set(range(first_token, last_token))
        span1 = set(*self.indices)
        span2 = set(*other.indices)
        both_spans = span1 | span2
        intervening = all_tokens - both_spans
        intervening_span = (min(intervening), max(intervening))
        return Span.from_indices(intervening_span)  

    @classmethod
    def from_in_sentence_indices(cls, indices, sentence):
        if indices[0] >= 0 and indices[1] <= len(sentence):
            tokens = sentence.tokens[indices[0]:indices[1]]
            return Span(tokens)
    
    @classmethod
    def from_indices(cls, indices, example)
        if indices[0] >= 0 and indices[1] <= len(example):
            tokens = example.tokens[indices[0]:indices[1]]
            return Span(tokens)


    def subspans(self):
        if len(self) == 1:
            return set()
        else:
            left_subspan_indices = (self.in_sentence_indices[0] + 1, self.in_sentence_indices[1])
            right_subspan_indices = (self.in_sentence_indices[0], self.in_sentence_indices[1] - 1)

            left_subspan = self.from_in_sentence_indices(left_subspan_indices, self.parent_sentence)
            right_subspan = self.from_in_sentence_indices(right_subspan_indices, self.parent_sentence)

            return set(left_subspan, right_subspan)

    def superspans(self):
        superspans = set()
        
        left_superspan_indices = (self.in_sentence_indices[0] - 1, self.in_sentence_indices[1])
        right_superspan_indices = (self.in_sentence_indices[0], self.in_sentence_indices[1] + 1)

        if left_superspan_indices[0] >= 0:
            left_superspan = Span(left_superspan_indices)
            superspans.add(left_superspan)
        if right_superspan_indices[1] <= len(self.parent_sentence):
            right_superspan = Span(right_superspan_indices)
            superspans.add(right_superspan)

        return superspans

    def negative_spans(self):
        return self.subspans() | self.superspans()          

    def __post_init__(self):
        self._get_in_sentence_indices()
        self._get_indices()
        self._get_parent_sentence()
        self._get_string()
        self._get_subword_indices()

#%% Sentence
@dataclass
class Sentence:
    tokens: list[Token]
    index: int
    parent_example: Example

    def candidate_spans(self, max_length):
        spans = set()
        for el1, el2 in combinations(self.tokens, 2):
            span = Span((el1.index, el2.index))
            if len(span) <= max_length:
                spans.add(span)

        return spans

    def __len__(self):
        return len(self.tokens)
#%% SpanPair
@dataclass
class SpanPair:
    span1: Span
    span2: Span
    parent_example: Example
    coref: Optional[int]

    def levenshtein_distance(self):
        return self.span1.lenvenshtein_distance(self.span2)

    def length_difference(self):
        return abs(len(self.span1.string) - len(self.span2.string))

    def token_distance(self):
        return self.span1.token_distance(self.span2)

    def sentence_distance(self):
        return self.span1.sentence_distance(self.span2)

    def intervening_span(self):
        return self.span1.intervening_span(self.span2)


    # def __hash__(self):
    #     return hash((span1, span2))

#%% ClassConverter
@dataclass
class ClassConverter:
    classes: list[str]

    def __post_init__(self):
        self.class2index = dict(zip(self.classes, range(len(self.classes))))
        self.index2class = dict(zip(range(len(self.classes)), self.classes))

#%% Cluster
@dataclass
class Cluster:
    spans: set[Span]
    type: Optional[str]
    parent_example: Example
    class_converter: Optional[ClassConverter]

    def __len__(self):
        return len(spans)
    
    def __hash__(self):
        return hash(spans)

    def pos_span_pairs(self):
       return [SpanPair(el1, el2) for el1, el2 in combinations(self.spans, 2)]

    def neg_span_pairs_cluster(self, other):
        return [SpanPair(el1, el2) for el1, el2 in product(self.spans, other.spans)]

    def neg_span_pairs(self, others):
        return unlist([self.neg_span_pairs_cluster(el) for el in others])

    def __post_init__(self):
        # self.class_index = self.class_converter.class2index[self.type]
        self.class_index = self.parent_example.parent_dataset.entity_type_converter.class2index[self.type]

#%% ClusterPair
@dataclass
class ClusterPair:
    head: Cluster
    tail: Cluster
    parent_example: Example

    relation_type: Optional[str]

    def __hash__(self):
        return hash((head, tail, relation_type))

    def relation_type_negatives(self):
        '''This will be too many negatives. Want to filter out less common relation_types somehow'''
        
        relation_types = class_converter.class2ix.keys()
        return set(ClusterPair(self.head, self.tail, el) for el in relation_types)

    def cluster_negatives_cluster(self, cluster, head_or_tail):
        if head_or_tail == 'head':
            return ClusterPair(cluster, self.tail, self.relation_type)
        elif head_or_tail == 'tail':
            return ClusterPair(self.head, cluster, self.relation_type)

    def cluster_negatives(self):
        clusters = self.parent_example.clusters - set([self.head, self.tail])
        hard_head_mutations = set(self.cluster_negatives_cluster(el, 'head') for el in clusters if self.el.class == self.head.class)
        hard_tail_mutations = set(self.cluster_negatives_cluster(el, 'tail') for el in clusters if self.el.class == self.tail.class)
        
        return hard_head_mutations | hard_tail_mutations

    def negative_cluster_pairs(self):
        negative_pairs = set()
        if self.parent_example.parent_dataset.relation_class_converter:
            negative_pairs.update(self.relation_type_negatives())
        negative_pairs.update(self.cluster_negatives())
        return list(negative_pairs)

    def enumerate_span_pairs(self):
        return [SpanPair(el1, el2) for el1, el2 in product(self.head.spans, self.tail.spans)]

#%% Example
@dataclass
class Example:
        
    # general
    sentences: list[Sentence]

    ### NER
    # mentions: list[Span]
    
    ### Cluster
    clusters: set[Cluster]

    ### RC
    positive_cluster_pairs: list[ClusterPair]

    ### other
    parent_dataset: Dataset

    def _get_mentions(self):
        self.mentions = set()
        for el in self.clusters:
            self.mentions.update(el.spans)

    def _get_negative_spans(self):
        negative_span_list = [el.negative_spans() for el in self.mentions] 
        self.negative_spans = set.union(*negative_span_list)
        
    def _get_candidate_spans(self):
        self.candidate_spans = set()
        for el in self.sentences:
            self.candidate_spans.add(el.candidate_spans())

    def _get_positive_span_pairs(self):
        self.positive_span_pairs = unlist([el.pos_span_pairs() for el in self.clusters()])
        
    def _get_negative_span_pairs(self):
        self.negative_span_pairs = unlist([el.neg_span_pairs() for el in self.clusters()])

    def _get_negative_cluster_pairs(self):
        self.negative_cluster_pairs = unlist([el.negative_cluster_pairs() for el in self.positive_cluster_pairs])

    def __len__(self):
        return len(self.tokens)

    def _get_tokens(self):
        self.tokens = unlist([el.tokens for el in self.sentences])

    def _get_subword_tokens(self):
        self.subword_tokens = unlist([el.subword_tokens for el in self.tokens])

    def __post_init__(self):
        self._get_tokens()
        # self._get_subword_tokens()
        self._get_mentions()
        self._get_negative_spans()
        self._get_candidate_spans()
        self._get_positive_span_pairs()
        self._get_negative_span_pairs()
        self._get_negative_cluster_pairs()
        self._get_subword_tokens()

#%% Dataset
@dataclass
class Dataset:

    examples: list[Example]
    tokenizer: ...
    entity_types: Optional[list[str]]
    relation_types: Optional[list[str]]

    def _get_type_converter(self, types):
        types.append('NA')
        return ClassConverter(types)

    def __post_init__(self):
        # (a) entity types don't exist
        # (b) entity types
        if self.entity_types:
            self.entity_type_converter = self._get_type_converter(self.entity_types)
        if self.relation_types:
            self.relation_type_converter = self._get_type_converter(self.relation_types)





#%% run
data_input_dir = 'data/processed/DocRED'
data_output_dir = 'data/modeling/DocRED'

make_dir(data_output_dir)


entity_types = torch.load(os.path.join(data_input_dir, 'entity_types.save'))
relation_types = torch.load(os.path.join(data_input_dir, 'relation_types.save'))
train_data = torch.load(os.path.join(data_input_dir, 'train_data.save'))

