#%% libraries
from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy

from spacy_data_classes import Span, EvalMention, SpanPair, EvalSpanPair, SpanGroup, EvalEntity, Relation, EvalRelation
from utils import defaultdict2dict

# from data_classes import ClusterPair
#%% Base Scorer
@dataclass
class Scorer:

    predicted: set
    gold: set
    
    def get_eval_objects(self, instance):
        if isinstance(instance, list):
            return set(self.get_eval_objects(el) for el in instance)
        if isinstance(instance, set):
            return set(self.get_eval_objects(el) for el in instance)
        if isinstance(instance, Span):
            return EvalMention.from_span(instance)
        if isinstance(instance, SpanPair):
            return EvalSpanPair.from_span_pair(instance)
        if isinstance(instance, SpanGroup):
            return EvalEntity.from_entity(instance)
        if isinstance(instance, Relation):
            return EvalRelation.from_relation(instance)

    def _make_set(self):
        self.predicted = set(self.predicted)
        self.gold = set(self.gold)

    def _copy(self):
        self.predicted = deepcopy(self.predicted)
        self.gold = deepcopy(self.gold)
        

    def __post_init__(self):
        self._copy()
        self._make_set()

    
        
        
   
#%% Base Uniclass Scorers
@dataclass
class UniclassScorer(Scorer):

    def score(self):
        # overall
        self.TP_objects = self.gold & self.predicted
        self.FP_objects = self.predicted - self.gold
        self.FN_objects = self.gold - self.predicted

        self.TP = len(self.TP_objects)
        self.FP = len(self.FP_objects)
        self.FN = len(self.FN_objects)

    def performance_counts(self):
        return {'global': {'TP': self.TP, 'FP': self.FP, 'FN': self.FN}}

#%% Base Multiclass Scorer
@dataclass
class MulticlassScorer(Scorer):

    # @staticmethod
    # def defaultdict_with_factory(factory):
    #     def inner_defaultdict():
    #         return defaultdict(factory)
    #     return inner_defaultdict
    
    # @staticmethod
    # def zero_factory():
    #     return 0

    # @staticmethod
    # def set_factory():
    #     return set()
    
    # @classmethod
    # def init_defaultdict(cls, factory):
    #     return defaultdict(cls.defaultdict_with_factory(factory))

    def get_multiclass_scores(self):
        # multiclass
        multiclass_objects = defaultdict(lambda: defaultdict(set))
        multiclass_scores = defaultdict(lambda: defaultdict(lambda: 0))
        
        for el in self.TP_objects:
            multiclass_objects[el.type]['TP'].add(el)
            multiclass_scores[el.type]['TP'] += 1
        for el in self.FP_objects:
            multiclass_objects[el.type]['FP'].add(el)
            multiclass_scores[el.type]['FP'] += 1
        for el in self.FN_objects:
            multiclass_objects[el.type]['FN'].add(el)
            multiclass_scores[el.type]['FN'] += 1

        multiclass_objects['global']['TP_objects'] = self.TP_objects
        multiclass_objects['global']['FP_objects'] = self.FP_objects
        multiclass_objects['global']['FN_objects'] = self.FN_objects
        
        multiclass_scores['global']['TP'] = self.TP
        multiclass_scores['global']['FP'] = self.FP
        multiclass_scores['global']['FN'] = self.FN

        self.multiclass_objects = defaultdict2dict(multiclass_objects)
        self.multiclass_scores = defaultdict2dict(multiclass_scores)

    def score(self):
        # overall
        self.TP_objects = self.gold & self.predicted
        self.FP_objects = self.predicted - self.gold
        self.FN_objects = self.gold - self.predicted

        self.TP = len(self.TP_objects)
        self.FP = len(self.FP_objects)
        self.FN = len(self.FN_objects)

        self.get_multiclass_scores()

    def __post_init__(self):
        super().__post_init__()
        self.score()

    def performance_counts(self):
        return self.multiclass_scores

#%% Base RelaxedScorer
@dataclass
class RelaxedScorer(MulticlassScorer):
    def _equal_type(self, predicted, gold):
        return predicted.type == gold.type
    
    def _entity_equality(self, predicted_entity, gold_entity):
        intersection = predicted_entity.mentions & gold_entity.mentions
        return len(intersection) > len(predicted_entity)/2

    def _relation_equality(self, predicted_relation, gold_relation):
        if not self._entity_equality(predicted_relation.head, gold_relation.head):
            return False
        if not self._entity_equality(predicted_relation.tail, gold_relation.tail):
            return False
        return True
    
    def _predicted_entity_in_gold(self, predicted_entity, is_typed):
        for el in self.gold:
            if is_typed:
                if self._entity_equality(predicted_entity, el) and self._equal_type(predicted_entity, el):
                    return el
            else:
                if self._entity_equality(predicted_entity, el):
                    return el
        return False

    def _predicted_relation_in_gold(self, predicted_relation, is_typed):
        for el in self.gold:
            if is_typed:
                if self._relation_equality(predicted_relation, el) and self._equal_type(predicted_relation, el):
                    return el
            else:
                if self._relation_equality(predicted_relation, el):
                    return el
        return False
        
    def score(self, is_typed, entity_or_relation):

        self.TP_objects = set()
        self.FP_objects = set()
        self.FN_objects = set()

        self.TP = 0
        self.FP = 0
        
        extracted_golds = set()

        for el in self.predicted:
            if entity_or_relation == 'entity':
                membership = self._predicted_entity_in_gold(el, is_typed)
            elif entity_or_relation == 'relation':
                membership = self._predicted_relation_in_gold(el, is_typed)
            if membership:
                if membership not in extracted_golds:
                    self.TP_objects.add(el)
                    self.TP += 1
                    extracted_golds.add(membership)
                elif membership in extracted_golds:
                    self.TP_objects.add(el)
            else:
                self.FP_objects.add(el)
                self.FP += 1
        
        self.FN_objects = self.gold - extracted_golds
        self.FN = len(self.FN_objects)

        self.get_multiclass_scores()

    def __post_init__(self):
        self._copy()
        self._make_set()


    




#%% NER Scorer
@dataclass
class TypedLocationNERScorer(MulticlassScorer):
    def __post_init__(self):
        super().__post_init__()
        self.score()

# @dataclass
# class TypedStringNERScorer(MulticlassScorer):
#     def __post_init__(self):
#         super().__post_init__()
#         self.score()           

# @dataclass
# class UntypedLocationNERScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'location')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'location')        
#         self.score()

# @dataclass
# class UntypedStringNERScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, True, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, True, 'string')        
#         self.score() 


    



#%% Coref scorer
@dataclass
class TypedLocationCorefScorer(UniclassScorer):
    
    def __post_init__(self):
        super().__post_init__()
        self.score()

#%% Entity Scorer
@dataclass
class TypedLocationStrictEntityScorer(MulticlassScorer):
    def __post_init__(self):
        super().__post_init__()      
        self.score()

# @dataclass
# class TypedStringStrictClusterScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, True, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, True, 'string')        
#         self.score()

# ##
# @dataclass
# class UntypedLocationStrictClusterScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'location')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'location')
#         self.score()

# #
# @dataclass
# class UntypedStringStrictClusterScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'string')
#         self.score()

            
@dataclass
class TypedLocationRelaxedEntityScorer(RelaxedScorer):
    
    def __post_init__(self):
        super().__post_init__()
        self.score(True, 'entity')

# @dataclass
# class TypedStringRelaxedClusterScorer(RelaxedScorer):
    
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, True, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, True, 'string')
#         self.score(True, 'cluster')

# @dataclass
# class UntypedLocationRelaxedClusterScorer(RelaxedScorer):
    
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'location')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'location')
#         self.score(False, 'cluster')

# @dataclass
# class UntypedStringRelaxedClusterScorer(RelaxedScorer):
    
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'string')
#         self.score(False, 'cluster')

#%% RC Scorers
@dataclass
class TypedLocationStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        super().__post_init__()
        self.score()

# #
# @dataclass
# class TypedStringStrictRCScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, True, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, True, 'string')
#         self.score()

##
@dataclass
class UntypedLocationStrictRCScorer(MulticlassScorer):
    
    def detype(self):
        gold = set()
        for el in self.gold:
            el.type = None
            gold.add(el)

        predicted = set()
        for el in self.predicted:
            el.type = None
            predicted.add(el)
            
    
    def __post_init__(self):
        super().__post_init__()
        self.detype()
        self.score()

# #
# @dataclass
# class UntypedStringStrictRCScorer(MulticlassScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'string')
#         self.score()


@dataclass
class TypedLocationRelaxedRCScorer(RelaxedScorer):
    def __post_init__(self):
        super().__post_init__()
        self.score(True, 'relation')

# @dataclass
# class TypedStringRelaxedRCScorer(RelaxedScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, True, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, True, 'string')
#         self.score(True, 'relation')

# @dataclass
# class UntypedLocationRelaxedRCScorer(RelaxedScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'location')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'location')
#         self.score(True, 'relation')

# @dataclass
# class UntypedStringRelaxedRCScorer(RelaxedScorer):
#     def __post_init__(self):
#         self.gold = self.get_eval_objects(self.gold, False, 'string')
#         self.predicted = self.get_eval_objects(self.predicted, False, 'string')
#         self.score(True, 'relation')








    

