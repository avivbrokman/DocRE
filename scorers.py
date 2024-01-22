#%% libraries
from dataclasses import dataclass, field, replace
from collections import defaultdict

from utils import generalized_replace
from data_classes import Span, SpanPair, Cluster, ClusterPair
# from data_classes import ClusterPair
#%% Base Scorer
@dataclass
class Scorer:

    predicted: set
    gold: set
    
    def get_eval_objects(self, instance, typed_eval, location_or_string_eval):
        if isinstance(instance, list):
            return [self.get_eval_objects(el, typed_eval, location_or_string_eval) for el in instance]
        if isinstance(instance, set):
            return set(self.get_eval_objects(el, typed_eval, location_or_string_eval) for el in instance)
        if isinstance(instance, (Span, SpanPair, Cluster, ClusterPair)):
            return instance.eval_copy(typed_eval, location_or_string_eval)
        
        
   

#%% Base Uniclass Scorers
@dataclass
class UniClassScorer(Scorer):

    def score(self):
        # overall
        self.TP_objects = self.gold & self.predicted
        self.FP_objects = self.predicted - self.gold
        self.FN_objects = self.gold - self.predicted

        self.TP = len(self.TP_objects)
        self.FP = len(self.FP_objects)
        self.FN = len(self.FN_objects)

        return {'TP': self.TP, 'FP': self.FP, 'FN': self.FN}

#%% Base Multiclass Scorer
@dataclass
class MulticlassScorer(Scorer):

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
        
        multiclass_objects['global']['TP'] = self.TP
        multiclass_objects['global']['FP'] = self.FP
        multiclass_objects['global']['FN'] = self.FN

        self.multiclass_objects = multiclass_objects
        self.multiclass_scores = multiclass_scores

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
        self.score()

    def performance_counts(self):
        return self.multiclass_scores

#%%
@dataclass
class RelaxedScorer(MulticlassScorer):
    def _equal_type(self, predicted, gold):
        return predicted.type == gold.type
    
    def _cluster_equality(self, predicted_cluster, gold_cluster):
        intersection = predicted_cluster.spans & gold_cluster.spans
        return len(intersection) > len(predicted_cluster)/2

    def _relation_equality(self, predicted_relation, gold_relation):
        if not self._cluster_equality(predicted_relation.head, gold_relation.head):
            return False
        if not self._cluster_equality(predicted_relation.tail, gold_relation.tail):
            return False
        return True
    
    def _predicted_relation_in_gold(self, predicted_relation, is_typed):
        for el in self.gold:
            if is_typed:
                if not self._equal_type(predicted_relation, el):
                    return False
            else:
                if self._relation_equality(predicted_relation, el):
                    return el
        return False
        
    def score(self, is_typed, cluster_or_relation):

        self.TP_objects = set()
        self.FP_objects = set()
        self.FN_objects = set()

        self.TP = 0
        self.FP = 0
        
        extracted_golds = set()

        for el in self.predicted:
            if cluster_or_relation == 'cluster':
                membership = self._predicted_cluster_in_gold(is_typed)
            elif cluster_or_relation == 'relation':
                membership = self._predicted_relation_in_gold(is_typed)
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
        self.score()


    




#%% NER Scorer
@dataclass
class TypedLocationNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'location')
        self.predicted = self.get_eval_objects(self.predicted, True, 'location')
        self.score()

@dataclass
class TypedStringNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')
        self.score()            

@dataclass
class UntypedLocationNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'location')
        self.predicted = self.get_eval_objects(self.predicted, False, 'location')        
        self.score()

@dataclass
class UntypedStringNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')        
        self.score() 


    



#%% coref



#%% Cluster Scorer
@dataclass
class TypedLocationStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'location')
        self.predicted = self.get_eval_objects(self.predicted, True, 'location')        
        self.score()

@dataclass
class TypedStringStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')        
        self.score()

##
@dataclass
class UntypedLocationStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'location')
        self.predicted = self.get_eval_objects(self.predicted, False, 'location')
        self.score()

#
@dataclass
class UntypedStringStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'string')
        self.predicted = self.get_eval_objects(self.predicted, False, 'string')
        self.score()

            


@dataclass
class TypedLocationRelaxedClusterScorer(RelaxedScorer):
    
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'location')
        self.predicted = self.get_eval_objects(self.predicted, True, 'location')
        self.score(True, 'cluster')

@dataclass
class TypedStringRelaxedClusterScorer(RelaxedScorer):
    
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')
        self.score(True, 'cluster')

@dataclass
class UntypedLocationRelaxedClusterScorer(RelaxedScorer):
    
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'location')
        self.predicted = self.get_eval_objects(self.predicted, False, 'location')
        self.score(False, 'cluster')

@dataclass
class UntypedStringRelaxedClusterScorer(RelaxedScorer):
    
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'string')
        self.predicted = self.get_eval_objects(self.predicted, False, 'string')
        self.score(False, 'cluster')

#%% RC Scorers
@dataclass
class TypedLocationStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'location')
        self.predicted = self.get_eval_objects(self.predicted, True, 'location')
        self.score()

#
@dataclass
class TypedStringStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')
        self.score()

##
@dataclass
class UntypedLocationStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'location')
        self.predicted = self.get_eval_objects(self.predicted, False, 'location')
        self.score()

#
@dataclass
class UntypedStringStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'string')
        self.predicted = self.get_eval_objects(self.predicted, False, 'string')
        self.score()


@dataclass
class TypedLocationRelaxedRCScorer(RelaxedScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'location')
        self.predicted = self.get_eval_objects(self.predicted, True, 'location')
        self.score(True, 'relation')

@dataclass
class TypedStringRelaxedRCScorer(RelaxedScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, True, 'string')
        self.predicted = self.get_eval_objects(self.predicted, True, 'string')
        self.score(True, 'relation')

@dataclass
class UntypedLocationRelaxedRCScorer(RelaxedScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'location')
        self.predicted = self.get_eval_objects(self.predicted, False, 'location')
        self.score(True, 'relation')

@dataclass
class UntypedStringRelaxedRCScorer(RelaxedScorer):
    def __post_init__(self):
        self.gold = self.get_eval_objects(self.gold, False, 'string')
        self.predicted = self.get_eval_objects(self.predicted, False, 'string')
        self.score(True, 'relation')








    

