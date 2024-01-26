#%% libraries
from dataclasses import dataclass, replace
from Levenshtein import distance
from typing import Optional
from itertools import combinations, product
from torch import Tensor

from utils import make_dir, unlist, parentless_print

#%% Token
@parentless_print
@dataclass
class Token:
    string: str
    in_sentence_index: int
    index: int
    # tokenizer_indices: int
    parent_sentence: ...

    def __len__(self):
        return len(self.subword_tokens)

    def _get_subword_tokens(self):
        tokenizer = self.parent_sentence.parent_example.parent_dataset.tokenizer
        self.subword_tokens = tokenizer.encode(self.string, add_special_tokens = False)

    def _previous_in_sentence_subword_length(self):
        return sum(len(el) for el in self.parent_sentence[0:self.in_sentence_index])    

    @classmethod        
    def _sentence_subword_length(cls, sentence):
        return sum(len(el) for el in sentence) 
    
    def _previous_subword_length(self):

        previous_sentences = self.parent_sentence.parent_example.sentences[0:self.parent_sentence.index]
        previous_sentence_lengths = sum(Token._sentence_subword_length(el) for el in previous_sentences)
        previous_length = previous_sentence_lengths + self._previous_in_sentence_subword_length()
        return previous_length

    
    def _get_subword_indices(self):  
        start = self._previous_subword_length()
        end = start + len(self) 
        self.subword_indices = (start, end)

    def __post_init__(self):
        self._get_subword_tokens()
        self._get_subword_indices()
#%% Span
@parentless_print
@dataclass
class Span:
    tokens: list[Token]
    type: Optional[str] = None
    parent_sentence: ... = None

    typed_eval: bool = True
    location_or_string_eval: str = 'location'

    def __hash__(self):
        hash_list = list()
        if self.location_or_string_eval == 'location':
            hash_list.append(self.indices)
        elif self.location_or_string_eval == 'string':
            hash_list.append(self.string)
        
        if self.typed_eval:
            hash_list.append(self.type)
        
        hash_tuple = tuple(hash_list)

        return hash(hash_tuple)

    def __eq__(self, other):
        if self.typed_eval:
            if self.type != other.type:
                return False
        if self.location_or_string_eval == 'location':
            return self.indices == other.indices
        elif self.location_or_string_eval == 'string':
            return self.string == other.string
        
        # return hash(self) == hash(other)

    def __len__(self):
        return len(range(*self.indices))
    
    def __bool__(self):
        return bool(self.indices)

    def eval_copy(self, typed_eval, location_or_string_eval):
        
        eval_copy = Span(None, self.type, None, typed_eval, location_or_string_eval)
        eval_copy.string = self.string
        eval_copy.indices = self.indices

        return eval_copy

        

    def _get_in_sentence_indices(self):
        self.in_sentence_indices = (self.tokens[0].in_sentence_index, self.tokens[-1].in_sentence_index + 1)

    def _get_indices(self):
        self.indices = (self.tokens[0].index, self.tokens[-1].index + 1)    

    def _get_parent_sentence(self):
        self.parent_sentence = self.tokens[0].parent_sentence

    def _get_string(self):
        self.string = ' '.join([el.string for el in self.tokens]) 

    def _get_subword_indices(self):
        self.subword_indices = (self.tokens[0].subword_indices[0], self.tokens[-1].subword_indices[1])

    def levenshtein_distance(self, other):
        return distance(self.string, other.string)

    def token_distance(self, other):
        first_token = min(self.indices[0], other.indices[0])
        last_token = max(self.indices[1], other.indices[1])
        all_tokens = set(range(first_token, last_token))
        span1 = set(range(*self.indices))
        span2 = set(range(*other.indices))
        both_spans = span1 | span2
        intervening = all_tokens - both_spans
        return len(intervening)

    def sentence_distance(self, other):
        return abs(other.parent_sentence.index - self.parent_sentence.index)

    def intervening_span(self, other):
        first_token = min(self.indices[0], other.indices[0])
        last_token = max(self.indices[1], other.indices[1])
        all_tokens = set(range(first_token, last_token))
        span1 = set(range(*self.indices))
        span2 = set(range(*other.indices))
        both_spans = span1 | span2
        intervening = all_tokens - both_spans
        if intervening:
            intervening_span = (min(intervening), max(intervening) + 1)
            try:
                intervening_span = Span.from_indices(intervening_span, self.parent_sentence.parent_example)
            except:
                intervening_span = Span.from_indices(intervening_span, self.parent_example)
            intervening_span._get_subword_indices()
            return intervening_span
        else:
            stub = Span.stub()
            stub.indices = None #(100000, 100000 + 1)
            stub.subword_indices = None #(100000, 100000 + 1)
            return stub 

    @classmethod
    def from_in_sentence_indices(cls, indices, sentence):
        if indices[0] >= 0 and indices[1] <= len(sentence):
            tokens = sentence.tokens[indices[0]:indices[1]]
            span = Span(tokens)
            span.process()
            return span
    
    @classmethod
    def from_indices(cls, indices, example):
        # if indices[0] >= 0 and indices[1] <= len(example.tokens) + 1:
        tokens = example.tokens[indices[0]:indices[1]]
        span = Span(tokens)
        span.in_sentence_indices = None
        span.indices = indices
        span._get_string()
        span.parent_example = example
        try:
            span._get_subword_indices()
        except:
            pass
        return span
        
    def typed_copy(self, type):
        span = Span.from_indices(self.indices, self.parent_sentence.parent_example)
        span.parent_sentence = self.parent_sentence
        span.type = type
        return span
    
    def detach(self):
        self.parent_sentence = None

    def subspans(self):
        if len(self) == 1:
            return set()
        else:
            left_subspan_indices = (self.in_sentence_indices[0] + 1, self.in_sentence_indices[1])
            right_subspan_indices = (self.in_sentence_indices[0], self.in_sentence_indices[1] - 1)

            left_subspan = self.from_in_sentence_indices(left_subspan_indices, self.parent_sentence)
            right_subspan = self.from_in_sentence_indices(right_subspan_indices, self.parent_sentence)

            return {left_subspan, right_subspan}

    def superspans(self):
        superspans = set()
        
        left_superspan_indices = (self.in_sentence_indices[0] - 1, self.in_sentence_indices[1])
        right_superspan_indices = (self.in_sentence_indices[0], self.in_sentence_indices[1] + 1)

        if left_superspan_indices[0] >= 0:
            left_superspan = Span.from_in_sentence_indices(left_superspan_indices, self.parent_sentence)
            superspans.add(left_superspan)
        if right_superspan_indices[1] <= len(self.parent_sentence):
            right_superspan = Span.from_in_sentence_indices(right_superspan_indices, self.parent_sentence)
            superspans.add(right_superspan)

        return superspans

    def negative_spans(self):
        return self.subspans() | self.superspans()          

    def process(self):
        self._get_in_sentence_indices()
        self._get_indices()
        self._get_parent_sentence()
        self._get_string()
        # self._get_subword_indices()

    @classmethod
    def stub(cls):
        return Span(list())

#%% Sentence
@parentless_print
@dataclass
class Sentence:
    tokens: list[Token]
    index: int
    parent_example: ...

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx]
    
    # def candidate_spans(self, max_length):
    def candidate_spans(self):
        spans = set()
        for el1, el2 in combinations(self.tokens, 2):
            span = Span.from_indices((el1.index, el2.index), self.parent_example)
            span.process()
            span._get_subword_indices()
            if len(span) <= self.parent_example.parent_dataset.max_length:
                spans.add(span)

        return spans
    
    @classmethod
    def stub(cls):
        return Sentence(list(), None, None)


#%% SpanPair
@parentless_print
@dataclass
class SpanPair:
    span1: Span
    span2: Span
    parent_example: ... = None
    coref: int = None

    def __hash__(self):
        hash_list = [self.span1.indices, self.span2.indices]
        hash_list.sort()
        hash_list.append(self.coref)
        return hash(tuple(hash_list))
    
    def __eq__(self, other):

        return {self.span1.indices, self.span2.indices, self.coref} == {other.span1.indices, other.span2.indices, other.coref}
            
    def __post_init__(self):
        self.spans = {self.span1, self.span2}
        self.type = self.span1.type

    def levenshtein_distance(self):
        return self.span1.levenshtein_distance(self.span2)

    def length_difference(self):
        return abs(len(self.span1.string) - len(self.span2.string))

    def token_distance(self):
        return self.span1.token_distance(self.span2)

    def sentence_distance(self):
        return self.span1.sentence_distance(self.span2)

    def intervening_span(self):
        return self.span1.intervening_span(self.span2)
    
    def eval_copy(self, typed_eval, location_or_string_eval):
        span1_copy = self.span1.eval_copy(typed_eval, location_or_string_eval)
        span2_copy = self.span2.eval_copy(typed_eval, location_or_string_eval)
        eval_copy = SpanPair(span1_copy, span2_copy, None, self.coref)
        return eval_copy

    # def does_overlap(self, other):
    #     self_strings = {self.span1.string, self.span2.string}
    #     other_strings = {other.span1.string, other.span2.string}

    #     return self_strings & other_strings
    #     # return bool(self_strings & other_strings)
    
    # def does_overlap2(self, cluster):
    #     cluster_strings = set(el.string for el in cluster.spans)
    #     pair_strings = {self.span1.string, self.span2.string}

    #     return pair_strings & 

    @classmethod
    def stub(cls):
        return SpanPair(None, None)
        

#%% ClassConverter
@parentless_print
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
#%% Cluster
@parentless_print
@dataclass
class Cluster:
    spans: set[Span]
    parent_example: ...

    type: Optional[str] = None
    class_converter: Optional[ClassConverter] = None

    typed_eval: bool = True
    location_or_string_eval: str = 'location'

    def __hash__(self):
        hash_list = sorted(self.spans, key = lambda span: hash(span))
        
        if self.typed_eval:            
            hash_list.append(self.type)
        
        hash_tuple = tuple(hash_list)
        
        return hash(hash_tuple)

    def __eq__(self, other):
        if self.typed_eval:
            if self.type != other.type:
                return False
        return self.spans == other.spans
    
    def __len__(self):
        return len(self.spans)

    def pos_span_pairs(self):
       return [SpanPair(el1, el2, self.parent_example, 1) for el1, el2 in combinations(self.spans, 2)]

    def neg_span_pairs_cluster(self, other):
        return [SpanPair(el1, el2, self.parent_example, 0) for el1, el2 in product(self.spans, other.spans)]

    def neg_span_pairs(self, others):
        return unlist([self.neg_span_pairs_cluster(el) for el in others])

    def process(self):
        # self.class_index = self.class_converter.class2index[self.type]
        self.class_index = self.parent_example.parent_dataset.entity_class_converter.class2index(self.type)

    def eval_copy(self, typed_eval, location_or_string_eval):
        eval_copy = Cluster({el.eval_copy(typed_eval, location_or_string_eval) for el in self.spans}, None, self.type, self.class_converter, typed_eval, location_or_string_eval)

        return eval_copy

    @classmethod
    def stub(cls):
        return Cluster(set(), None)
    
    def populate(self, spans, parent_example, type, class_converter):
        self = replace(self, spans = spans, parent_example = parent_example, type = type, class_converter = class_converter)

#%% ClusterPair
@parentless_print
@dataclass
class ClusterPair:
    head: Cluster
    tail: Cluster
    parent_example: ...

    type: Optional[str] = None

    typed_eval: bool = True
    location_or_string_eval: str = 'location'

    def __hash__(self):
        hash_list = [self.head, self.tail]
        if self.typed_eval:
            hash_list.append(self.type)
        hash_tuple = tuple(hash_list)
        
        return hash(hash_tuple)

    def __eq__(self, other):
        if self.typed_eval:
            if self.type != other.type:
                return False
        if self.head != other.head:
            return False
        if self.tail != other.tail:
            return False
        return True

    def _filter_positive_relations(self, cluster_pairs):
        return cluster_pairs - self.parent_example.positive_cluster_pairs

    def _null_relation(self):
        return ClusterPair(self.head, self.tail, None)

    def _relation_type_mutations(self):
        
        relation_types = self.parent_example.parent_dataset.relation_class_converter.class2index_dict.keys()
        mutated_relations_with_cluster_pair = set(ClusterPair(self.head, self.tail, self.parent_example, el) for i, el in enumerate(relation_types) if i < 10)
        negative_relations_with_cluster_pair = self._filter_positive_relations(mutated_relations_with_cluster_pair)
        return negative_relations_with_cluster_pair

    def _mutate_cluster_pair(self, cluster, head_or_tail):
        if head_or_tail == 'head':
            return ClusterPair(cluster, self.tail, self.parent_example, self.type)
        elif head_or_tail == 'tail':
            return ClusterPair(self.head, cluster, self.parent_example, self.type)

    def _cluster_mutations(self):
        clusters = self.parent_example.clusters - set([self.head, self.tail])
        head_mutations = set(self._mutate_cluster_pair(el, 'head') for el in clusters if el.type == self.head.type)
        tail_mutations = set(self._mutate_cluster_pair(el, 'tail') for el in clusters if el.type == self.tail.type)

        mutations = head_mutations | tail_mutations

        negative_relations = self._filter_positive_relations(mutations)
        
        return negative_relations

    def negative_cluster_pairs(self):
        negative_pairs = set()
        # negative_pairs.add(self._null_relation())
        negative_pairs.update(self._relation_type_mutations())
        negative_pairs.update(self._cluster_mutations())

        return list(negative_pairs)

    def enumerate_span_pairs(self):
        return [SpanPair(el1, el2) for el1, el2 in product(self.head.spans, self.tail.spans)]
    
    def eval_copy(self, typed_eval, location_or_string_eval):
        head = self.head.eval_copy(typed_eval, location_or_string_eval)
        tail = self.tail.eval_copy(typed_eval, location_or_string_eval)
        eval_copy = ClusterPair(head, tail, None, self.type, typed_eval, location_or_string_eval)

        return eval_copy

    @classmethod
    def stub(cls):
        return ClusterPair(None, None, None)


#%% Example
@parentless_print
@dataclass
class Example:
        
    # general
    sentences: list[Sentence]
    
    ### Cluster
    clusters: set[Cluster]

    ### RC
    positive_cluster_pairs: list[ClusterPair]

    ### other
    parent_dataset: ...

    def _get_mentions(self):
        self.mentions = set()
        for el in self.clusters:
            self.mentions.update(el.spans)

    def _get_negative_spans(self):
        negative_span_list = [el.negative_spans() for el in self.mentions] 
        self.negative_spans = set.union(*negative_span_list)
        self.negative_spans -= self.mentions
        
    def _get_candidate_spans(self):
        self.candidate_spans = set()
        self.candidate_spans.update(self.mentions)
        for el in self.sentences:
            self.candidate_spans.update(el.candidate_spans())

    def _get_positive_span_pairs(self):
        self.positive_span_pairs = unlist([el.pos_span_pairs() for el in self.clusters])
        
    def _get_negative_span_pairs(self):
        self.negative_span_pairs = unlist([el.neg_span_pairs(self.clusters - set([el])) for el in self.clusters])

    def _get_negative_cluster_pairs(self):
        self.negative_cluster_pairs = unlist([el.negative_cluster_pairs() for el in self.positive_cluster_pairs])

    def _get_candidate_cluster_pairs(self):
        self.candidate_cluster_pairs = [ClusterPair(el1, el2, self) for el1, el2 in combinations(self.clusters, 2)]

    def __len__(self):
        return len(self.tokens)

    def _get_tokens(self):
        self.tokens = unlist([el.tokens for el in self.sentences])

    def _get_subword_tokens(self):
        self.subword_tokens = unlist([el.subword_tokens for el in self.tokens])

    def process(self):
        self._get_tokens()
        # self._get_subword_tokens()
        self._get_mentions()
        self._get_negative_spans()
        self._get_candidate_spans()
        self._get_positive_span_pairs()
        self._get_negative_span_pairs()
        self._get_negative_cluster_pairs()
        self._get_candidate_cluster_pairs()
        self._get_subword_tokens()

    @classmethod
    def stub(cls):
        return Example(list(), set(), list(), None)
    
    def populate(self, sentences, clusters, positive_cluster_pairs, parent_dataset):
        self.sentences = sentences
        self.clusters = clusters
        self.positive_cluster_pairs = positive_cluster_pairs
        self.parent_dataset = parent_dataset

        self.process()

#%% Dataset 
@parentless_print
@dataclass
class Dataset:

    examples: list[Example]
    tokenizer: ...
    max_length: int
    entity_types: Optional[list[str]]
    relation_types: Optional[list[str]]

    def __len__(self):
        return len(self.examples)

    def _get_type_converter(self, types):
        types.append(None)
        return ClassConverter(types)

    def process(self):
        # (a) entity types don't exist
        # (b) entity types
        if self.entity_types:
            self.entity_class_converter = self._get_type_converter(self.entity_types)
        if self.relation_types:
            self.relation_class_converter = self._get_type_converter(self.relation_types)


    def __getitem__(self, idx):
    
        return self.examples[idx]

    @classmethod
    def stub(cls, tokenizer, max_length, entity_types, relation_types):
        return Dataset(list(), tokenizer, max_length, entity_types, relation_types)
    
    def populate(self, examples):
        self.examples = examples
        self.process()

    def _get_subword_indices(self, object_):
    
        if isinstance(object_, list) or isinstance(object_, set):
            for el in object_:
                self._get_subword_indices(el)
            # map(self._get_subword_indices, object_)
        if isinstance(object_, Dataset):
            self._get_subword_indices(object_.examples)
        if isinstance(object_, Example):
            self._get_subword_indices(object_.sentences)

            self._get_subword_indices(object_.mentions)
            self._get_subword_indices(object_.negative_spans)
            self._get_subword_indices(object_.candidate_spans)

        if isinstance(object_, Sentence):
            self._get_subword_indices(object_.tokens)
        if isinstance(object_, Token):
            object_._get_subword_indices()
        if isinstance(object_, Span):
            # if not hasattr(object_, 'subword_indices'):
            object_._get_subword_indices()