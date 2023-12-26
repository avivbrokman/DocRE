#%% Reminder!  Deal with predictions, in terms of how spans are constructed, i.e. .type =.


#%% libraries
from dataclasses import dataclass, field, replace
from collections import defaultdict

from data_classes import Relation
#%% Base Scorer
@dataclass
def Scorer:

    predicted: set
    gold: set

    def adjust_span_mode(self, span, typed_mode, location_or_string_mode):
        
        return replace(span, type_mode = typed_mode, location_or_string_mode = location_or_string_mode)

    def adjust_spans_mode(self, spans, typed_mode, location_or_string_mode):
        
        return set(self.adjust_span_mode(el, typed_mode, location_or_string_mode) for el in spans)
        
    def adjust_cluster_mode(self, cluster, typed_mode, location_or_string_mode, strict_or_relaxed_mode):

        spans = self.adjust_spans_mode(cluster.spans)
        
        return replace(cluster, spans = spans, type_mode = typed_mode, location_or_string_mode = location_or_string_mode, strict_or_relaxed_mode = strict_or_relaxed_mode)

    def adjust_clusters_mode(self, clusters, typed_mode, location_or_string_mode, strict_or_relaxed_mode):
        
        return set(self.adjust_cluster_mode(el, typed_mode, location_or_string_mode, strict_or_relaxed_mode) for el in clusters)

    def adjust_relation_mode(self, relation, typed_mode, location_or_string_mode, strict_or_relaxed_mode):

        head = self.adjust_cluster_mode(head, typed_mode, location_or_string_mode, strict_or_relaxed_mode)
        tail = self.adjust_cluster_mode(tail, typed_mode, location_or_string_mode, strict_or_relaxed_mode)

        return replace(relation, head = head, tail = tail, typed_mode = typed_mode, location_or_string_mode = location_or_string_mode, strict_or_relaxed_mode = strict_or_relaxed_mode)

    def adjust_relations_mode(self, relations, typed_mode, location_or_string_mode, strict_or_relaxed_mode):
        
        return set(self.adjust_relation_mode(el, typed_mode, location_or_string_mode, strict_or_relaxed_mode) for el in relations)

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

        return {'TP': TP, 'FP': FP, 'FN': FN}

#%% Base Multiclass Scorer
@dataclass
class MultiClassScorer(Scorer):

    def score(self):
        # overall
        self.TP_objects = self.gold & self.predicted
        self.FP_objects = self.predicted - self.gold
        self.FN_objects = self.gold - self.predicted

        self.TP = len(self.TP_objects)
        self.FP = len(self.FP_objects)
        self.FN = len(self.FN_objects)

        # multiclass
        multiclass_objects = defaultdict(defaultdict(set))
        multiclass_scores = defaultdict(default_dict(lambda: 0))
        
        for el in self.TP_objects:
            multiclass_objects[el.type]['TP'].add(el)
            multiclass_scores[el.type]['TP'] += 1
        for el in self.FP_objects:
            multiclass_objects[el.type]['FP'].add(el)
            multiclass_scores[el.type]['FP'] += 1
        for el in self.FN_objects:
            multiclass_objects[el.type]['FN'].add(el)
            multiclass_scores[el.type]['FN'] += 1

        multi_class_objects['global']['TP_objects'] = self.TP_objects
        multi_class_objects['global']['FP_objects'] = self.FP_objects
        multi_class_objects['global']['FN_objects'] = self.FN_objects
        
        multi_class_objects['global']['TP'] = self.TP
        multi_class_objects['global']['FP'] = self.FP
        multi_class_objects['global']['FN'] = self.FN



#%% NER Scorer
@dataclass
def TypedLocationNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_spans_mode('typed', 'location')
        self.score()

@dataclass
def TypedStringNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_spans_mode('typed', 'string')    
        self.score()            

@dataclass
def UntypedLocationNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_spans_mode('untyped', 'location')
        self.score()

@dataclass
def UntypedStringNERScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_spans_mode('untyped', 'string') 
        self.score() 

#%% Cluster Scorer
@dataclass
class TypedLocationStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('typed', 'location', 'strict')
        self.score()

@dataclass
class TypedLocationRelaxedClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('typed', 'location', 'relaxed')
        self.score()
#
@dataclass
class TypedStringStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('typed', 'string', 'strict')
        self.score()

@dataclass
class TypedStringRelaxedClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('typed', 'string', 'relaxed')
        self.score()

##
@dataclass
class UntypedLocationStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('untyped', 'location', 'strict')
        self.score()

@dataclass
class UntypedLocationRelaxedClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('untyped', 'location', 'relaxed')
        self.score()

#
@dataclass
class UntypedStringStrictClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('untyped', 'string', 'strict')
        self.score()

@dataclass
class UntypedStringRelaxedClusterScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_clusters_mode('untyped', 'string', 'relaxed')
        self.score()

#%% RC Scorers
@dataclass
class TypedLocationStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('typed', 'location', 'strict')
        self.score()

@dataclass
class TypedLocationRelaxedRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('typed', 'location', 'relaxed')
        self.score()
#
@dataclass
class TypedStringStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('typed', 'string', 'strict')
        self.score()

@dataclass
class TypedStringRelaxedRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('typed', 'string', 'relaxed')
        self.score()

##
@dataclass
class UntypedLocationStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('untyped', 'location', 'strict')
        self.score()

@dataclass
class UntypedLocationRelaxedRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('untyped', 'location', 'relaxed')
        self.score()

#
@dataclass
class UntypedStringStrictRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('untyped', 'string', 'strict')
        self.score()

@dataclass
class UntypedStringRelaxedRCScorer(MulticlassScorer):
    def __post_init__(self):
        self.adjust_cluster_pairs_mode('untyped', 'string', 'relaxed')
        self.score()





#%% NER Scorers
@dataclass
class TypedLocationNERScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location')
        self.score_mentions()

@dataclass
class UntypedLocationNERScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location')
        self.score_mentions()

@dataclass
class TypedStringNERScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string')
        self.score_mentions()

@dataclass
class UntypedStringNERScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string')
        self.score_mentions()

#%% Cluster Scorers
@dataclass
class StrictTypedLocationClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictTypedStringClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedLocationClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedStringClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class RelaxedTypedLocationClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedTypedStringClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedLocationClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedStringClusterScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

#%% RC scorers (DocRED, BioRED)
@dataclass
class StrictTypedLocationRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictTypedStringRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedLocationRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedStringRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class RelaxedTypedLocationRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedTypedStringRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedLocationRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedStringRCScorer(MulticlassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

#%% RC scorers (CDR)
@dataclass
class StrictTypedLocationRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictTypedStringRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedLocationRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class StrictUntypedStringRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'strict')
        self.score_mentions()

@dataclass
class RelaxedTypedLocationRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedTypedStringRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'typed', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedLocationRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'location', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

@dataclass
class RelaxedUntypedStringRCScorer(UniclassScorer):
        
    def __post_init__(self):
        self.adjust_mode(typed_mode = 'untyped', location_or_string_mode = 'string', strict_or_relaxed_mode = 'relaxed')
        self.score_mentions()

#%%


@dataclass
class TypedSpanNERScorer(NERScorer):
        
    def __post_init__(self):
        self.adjust_hash(typed_mode = 'typed', location_or_string_mode = 'span')
        self.score_mentions()

@dataclass
class UntypedSpanNERScorer(NERScorer):
        
    def __post_init__(self):
        self.adjust_hash(typed_mode = 'untyped', span_or_string_mode = 'span')
        self.score_mentions()

@dataclass
class UntypedNERScorer(NERScorer):

    def _adjust_hash(self):
        for el in self.predicted_mentions:
            replace(el, hash_mode = 'untyped')
        for el in self.gold_mentions:
            replace(el, hash_mode = 'untyped') 
        
    def __post_init__(self):
        self.adjust_hash()
        self.score_mentions()

#%% Base cluster scorer



    

