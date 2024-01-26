#%% libraries
from dataclasses import dataclass
from itertools import combinations, product
from torch import Tensor
from collections import Counter, defaultdict
from Levenshtein import distance
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex


from spacy.tokens import Token, Span, SpanGroup, Doc

from utils import unlist, mode, parentless_print


#%% modify tokenizer
def tokenize_on_special_characters(text_processor, special_characters):
    # Get the default infix patterns from the tokenizer
    infixes = text_processor.Defaults.infixes

    # Add custom infix patterns for the provided special characters
    custom_infixes = [re.escape(char) for char in special_characters]
    infixes = infixes + custom_infixes

    # Compile new infix regex
    infix_regex = compile_infix_regex(infixes)

    # Create a new tokenizer with the custom infix pattern
    text_processor.tokenizer = Tokenizer(text_processor.vocab, 
                              prefix_search=text_processor.tokenizer.prefix_search,
                              suffix_search=text_processor.tokenizer.suffix_search,
                              infix_finditer=infix_regex.finditer,
                              token_match=text_processor.tokenizer.token_match)

    return text_processor


#%% Modifying Spacy structures
Token.set_extension("subwords", default = None)
Token.set_extension("subword_indices", default = None)

Span.set_extension("id", default = None)
Span.set_extension("sentence_index", default = None)

Doc.set_extension("mentions", default = set())
Doc.set_extension("relations", default = set())

#%% SpanUtils
class SpanUtils:
    
    @classmethod
    def subword_indices(cls, span):
        return (span.subword_indices[0], span.subword_indices[1])
    
    @classmethod
    def subspans(cls, span):
        subspans = set()
        left_subspan = span[1:]
        right_subspan = span[:-1]
        if left_subspan:
            subspans.add(left_subspan)
        if right_subspan:
            subspans.add(right_subspan)

        return subspans
    
    @classmethod
    def superspans(cls, span, doc):
        superspans = set()
        if span.start != span.sent.start:
            left_superspan = doc[(span.start - 1):span.end]
            superspans.add(left_superspan)
        if span.end != span.sent.end:
            right_superspan = doc[span.start:(span.end + 1)]
            superspans.add(right_superspan)

        return superspans
    
    @classmethod
    def has_noun(cls, span):
        for token in span:
            if token.pos_ in ['NOUN', 'PROPN']:
                return True
        return False

#%% SpanPair
@dataclass
class SpanPair:
    span1: Span
    span2: Span
    coref: int = None
    
    def levenshtein_distance(self):
        return distance(self.span1.text, self.span2.text)
    
    def intervening_token_indices(self):
        first_token = min(self.span1.start, self.span2.start)
        last_token = max(self.span1.end, self.span2.end)
        all_tokens = set(range(first_token, last_token))
        span1 = set(range(self.span1.start, self.span1.start))
        span2 = set(range(self.spanw.start, self.spanw.start))
        both_spans = span1 | span2
        intervening = all_tokens - both_spans
        return intervening
    
    def length_difference(self):
        return abs(len(self.span1.text) - len(self.span2.text))
    
    def token_distance(self):
        return len(self.intervening_token_indices())

    def sentence_distance(self):
        return abs(self.span1.sent.sentence_index - self.span2.sent.sentence_index)

    def intervening_span(self):
        intervening = self.intervening_token_indices()
        if intervening:
            intervening_span = (min(intervening), max(intervening) + 1)
            intervening_span = self.span1.doc[range(*intervening_span)]
            return intervening_span
        
        # !!!!! WHAT SHOULD THE PLACEHOLDER BE !!!!!!!!
        else:
            stub = Span.stub()
            stub.indices = None #(100000, 100000 + 1)
            stub.subword_indices = None #(100000, 100000 + 1)
            return stub 
    
    
#%% Entity
class Entity(SpanGroup):
    
    def pos_span_pairs(self):
       return [SpanPair(el1, el2, 1) for el1, el2 in combinations(self.spans, 2)]

    def neg_span_pairs_entity(self, other):
        return [SpanPair(el1, el2, 0) for el1, el2 in product(self.spans, other.spans)]

    def neg_span_pairs(self, others):
        return unlist([self.neg_span_pairs_entity(el) for el in others])    

            
#%% Relation
@dataclass
class Relation:
    head: Entity
    tail: Entity
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
    
    def _filter_positive_relations(self, relations, positive_relations):
       return relations - positive_relations
    
    def is_valid(self, valid_combinations):
        combination = {self.head.type, self.tail.type, self.type}
        return combination in valid_combinations
    
    @classmethod
    def filter_invalid_relations(cls, relations, valid_combinations):
        return set(el for el in relations if el.is_valid(valid_combinations))
    
    def _mutate_relation_type(self, relation_type):
        return Relation(self.head, self.tail, relation_type)
    
    def relation_type_mutations(self, relation_types, positive_relations):
        #relation_types = relation_class_converter.class2index_dict.keys()
        mutated_relations = set(self._mutate_relation_type(el) for el in relation_types)
        negative_relations = self._filter_positive_relations(mutated_relations, positive_relations)
        return negative_relations

    def _mutate_entity(self, entity, head_or_tail):
        if head_or_tail == 'head':
            return Relation(entity, self.tail, self.type)
        elif head_or_tail == 'tail':
            return Relation(self.head, entity, self.type)

    def entity_mutations(self, entities):
        entities = entities - {self.head, self.tail}
        head_mutations = set(self._mutate_cluster(el, 'head') for el in clusters if el.type == self.head.type)
        tail_mutations = set(self._mutate_cluster(el, 'tail') for el in clusters if el.type == self.tail.type)

        mutations = head_mutations | tail_mutations

        negative_relations = self._filter_positive_relations(mutations)
        
        return negative_relations

    def negative_relations(self):
        negative_pairs = set()
        negative_pairs.update(self._relation_type_mutations())
        negative_pairs.update(self._cluster_mutations())

        return negative_pairs

    def enumerate_span_pairs(self):
        return [SpanPair(el1, el2) for el1, el2 in product(self.head.spans, self.tail.spans)]    
        

#%% Example
@parentless_print
@dataclass
class Example:

    annotation: ...
    parent_dataset: ...
    
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
        self.title = self.annotation['pmid']

    def _get_text(self):
        passages = self.annotation['passages']
        for el in passages:
            if el['type'] == 'title':
                self.title = el['text'][0]
            if el['type'] == 'abstract':
                self.abstract = el['text'][0]
        
        self.text = self.title + ' ' + self.abstract 
        
    def _get_doc(self):
        self.doc = self.parent_dataset.text_processor(self.text)
        
#     def _get_character_word_index_converters(self):
#         self.start_character2word = dict()
#         self.end_character2word = dict()
# # 
# #         self.word2character = dict()
# # 
#         
#         for el in self.doc.tokens:
#             start_character_index = el.idx
#             end_character_index = start_character_index + len(el)
#             word_index = el.i
#             
#             self.start_character2word[start_character_index] = word_index
#             self.end_character2word[end_character_index] = word_index + 1
    
    def _get_sentence_indices(self):
        for i, el in enumerate(self.doc.sents):
            el._.sentence_index = i

    def _get_mention(self, annotation):
# =============================================================================
#         start_character_index, end_character_index = annotation['offsets'][0]
#         start_word = self.start_character2word[start_character_index]
#         end_word = self.end_character2word[end_character_index]
#         span = self.doc[start_word:end_word]
# =============================================================================
        start_char_index, end_char_index = annotation['offsets'][0]
        type_ = annotation['semantic_type_id']
        span = self.doc.char_span(start_char_index, end_char_index, label = type_)
        span._.id = annotation['concept_id']
        
        return span
    
    def _get_mentions(self):
        self.doc._.mentions = set(self._get_mention(el) for el in self.annotation['entities'] if len(el['offsets']) == 1)
    
# =============================================================================
#     def _get_entities(self):
#         entity_dict = default_dict(set)
#         for el in self.annotation['entities']:
#             if len(el.offsets) == 1: # removes discontinuous mentions
#                 span = self._get_span(el)
#                 entity_dict[span._.id].add(span)
#         for id_, entity in entity_dict.items():
#             type_ = list(entity)[0]._.type
#             entity_dict[id_] = SpanGroup(self.doc, 
#                                          name = id_, 
#                                          attrs = {'id': id_, 'type': type_},
#                                          spans = entity
#                                          )
#         self.doc.spans = entity_dict
# =============================================================================
    
    def _get_entities(self):
        entity_dict = defaultdict(set)
        for el in self.doc._.mentions:
                entity_dict[el._.id].add(el)
        for id_, entity in entity_dict.items():
            type_ = list(entity)[0].label_
            self.doc.spans[id_] = Entity(self.doc, 
                                         name = id_,
                                         attrs = {'id': id_, 'type': type_},
                                         spans = entity
                                         )

    def _get_relation(self, annotation):
        entity1 = self.doc.spans[annotation['concept_1']]
        entity2 = self.doc.spans[annotation['concept_2']]
        relation = Relation(entity1, entity2, annotation['type'])
        return relation
    
        
    def _get_relations(self):
        
        self.doc._.relations = {self._get_relation(el) for el in self.annotation['relations']}
    
    def _tokenize(self):
        for i, el in enumerate(self.doc):
            el._.subwords = self.parent_dataset.tokenizer.encode(el.text, add_special_tokens = False) #el.text_with_ws?
            if i == 0:
                el._.subword_indices = (0, len(el._.subwords))
            else:
                start_index = self.doc[i - 1]._.subword_indices[1]
                end_index = start_index + len(el._.subwords)
                el._.subword_indices = (start_index, end_index)
                
    # for NER training            
    def _subspans(self):
        subspans = set()
        for entity in self.doc.spans:
            subspan_list = [SpanUtils.subspans(span) for span in entity.spans]
            subspans.update(set.union(*subspan_list))                
                    
        return subspans
    
    def _superspans(self):
        superspans = set()
        for entity in self.doc.spans:
            superspan_list = [SpanUtils.superspans(span, self.doc) for span in entity.spans]
            superspans.update(set.union(*superspan_list))           
                                
        return superspans
    
    def negative_spans(self):
        other_spans = self._subspans() | self._superspans()
        
        negative_spans = set()
        for el_other in other_spans:
            for el_mention in self.doc._.mentions:
                if el_other.start == el_mention.start and el_other.end == el_mention.end:
                    break
            else:
                negative_spans.append(el_other)
                
        return negative_spans

    # for NER inference
    def candidate_spans(self):
        candidate_spans = set()
        candidate_spans.update(self.doc._.mentions)
        mention_indices = {(el.start, el.end) for el in self.doc._.mentions}
        for sentence in self.doc.sents:
            for start in range(sentence.start, sentence.end - 1):
                for end in range(start + 1, sentence.end):
                    span = self.doc[start:end]
                    if (span.start, span.end) not in mention_indices:
                        candidate_spans.append(span)
            
        return candidate_spans
    
    # For Coref training
    def positive_span_pairs(self):
        return unlist([el.pos_span_pairs() for el in self.doc.spans.values()])
        
    def negative_span_pairs(self):
        entities = set(self.doc.spans.keys())
        return unlist([el.neg_span_pairs(entities - {el}) for el in entities])

    # For RC training
    def negative_relations(self):
        return unlist([el.negative_relations() for el in self.doc._.relations])

    # def candidate_relations(self):
    #     self.candidate_cluster_pairs = [Relation(el1, el2) for el1, el2 in combinations(self.clusters, 2)]
         

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
    
#%% BioREDDataset
@dataclass
class Dataset:
    
    huggingface_dataset: ...
    text_processor: ...
    tokenizer: ...

    entity_types: list[str] = None
    relation_types: list[str] = None

    def __len__(self):
        self.len(self.examples)
    
    def __getitem__(self, idx):

        return self.examples[idx]

    def _get_examples(self):
        self.examples = [Example(el, self) for el in self.huggingface_dataset]

    def _get_type_converter(self, types):
        types.append(None)
        return ClassConverter(types)

    def get_entity_types(self):
        counts = Counter()
        for el_ex in self.examples:
            for el_ent in el_ex.doc.spans.keys():
                counts.update([el_ent.attrs['type']])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        return list(types)
        
    def get_relation_types(self):
        counts = Counter()
        for el_ex in self.examples:
            for el_rel in el_ex._.relations:
                counts.update([el_rel.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        
        self.counts = counts

        return list(types) 

    def process(self):
        # (a) entity types don't exist
        # (b) entity types
        if self.entity_types:
            self.entity_class_converter = self._get_type_converter(self.entity_types)
        if self.relation_types:
            self.relation_class_converter = self._get_type_converter(self.relation_types)

    def __post_init__(self):
        self.process()
        self._get_examples()