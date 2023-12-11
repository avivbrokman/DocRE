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

@dataclass
class Token:
    string: str
    in_sentence_index: int
    index: int
    tokenizer: str
    tokenizer_indices: int
    parent_sentence: Sentence

@dataclass(frozen = True)
class Span:
    tokens: list[Token]
    class: Optional[str]

    def _get_in_sentence_indices(self):
        self.in_sentence_indices = (tokens[0].in_sentence_index, tokens[-1].in_sentence_index + 1)

    def _get_indices(self):
        self.indices = (tokens[0].index, tokens[-1].index + 1)    

    def _get_parent_sentence(self):
        self.parent_sentence = self.tokens[0].parent_sentence

    def _get_string(self):
        self.string = [el.string for el in self.tokens].join(' ')  

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

@dataclass
class Sentence:
    tokens: list[Token]
    index: int
    parent_example: Example

    def candidate_spans(self, max_length):
        for el in 

    def __len__(self):
        return len(self.tokens)

@dataclass
class SpanPair:
    span1: Span
    span2: Span
    parent_example: Example
    coref: Optional[int]

@dataclass
class ClassConverter:
    classes: list[str]

    def __post_init__(self):
        self.class2index = dict(zip(self.classes, range(len(self.classes))))
        self.index2class = dict(zip(range(len(self.classes)), self.classes))

@dataclass
class Cluster:
    spans: set[Span]
    class: Optional[str]
    parent_example: Example
    class_converter: Optional[ClassConverter]

    def __post_init__(self):
        self.class_index = self.class_converter.class2index[self.class]

@dataclass
class Example:
        
    # general
    sentences: list[Sentence]

    ### NER
    # train
    positive_spans: list[Span]
    
    # eval
    candidate_spans: list[Span]

    ### Cluster
    # train
    positive_span_pairs: list[SpanPair]
    negative_span_pairs: list[SpanPair]

    ### RC
    # train
    positive_entity_pairs: list[ClusterPair]
    negative_entity_pairs: list[ClusterPair]

    def _get_negative_spans(self):
        negative_span_list = [el.negative_spans() for el in self.positive_spans] 
        negative_spans = set.union(*negative_span_list)
        
        return negative_spans

    def _get_candidate_spans(self):
        

    def __len__(self):
        return len(self.tokens)

    def _get_tokens(self):
        self.tokens = unlist(el.tokens for el in self.sentences)

    def __post_init__(self):
        self._get_tokens()

@dataclass
class Dataset:

    examples: list[Example]





#####




#%% run
data_input_dir = 'data/processed/DocRED'
data_output_dir = 'data/modeling/DocRED'

make_dir(data_output_dir)

train_data = torch.load(os.path.join(data_input_dir, 'train_data.save'))

