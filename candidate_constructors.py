#%% libraries
from collections import defaultdict
from itertools import combinations, product

from data_classes_original import SpanPair, ClusterPair

#%% useful functions
def get_objects_by_type(objects):
    
    object_dict = defaultdict(set)

    for el in objects:
        object_dict[el.type].add(el)

    return object_dict


# #%% span_pair constructor
def exhaustive_intratype_pairs(spans):
    spans_by_type = get_objects_by_type(spans)
    
    span_pairs = set()
    for class_, class_spans in spans_by_type.items():
        span_pairs.update(set(SpanPair(el1, el2) for el1, el2 in combinations(class_spans, 2)))

    return span_pairs

#%% candidate_cluster_pair
def CDR_candidate_cluster_pair_constructor(clusters):
    
    chemicals = set(el for el in clusters if el.type == 'chemical')
    diseases = set(el for el in clusters if el.type == 'disease')

    cluster_pairs = set()
    cluster_pairs.update(set(ClusterPair(el1, el2) for el1, el2 in product(chemicals)))
    cluster_pairs.update(set(ClusterPair(el1, el2) for el1, el2 in product(diseases)))

    return cluster_pairs

def BioRED_candidate_cluster_pair_constructor(clusters):
    
    valid_type_combinations = set(('disease', 'chemical'),
                            ('disease', 'gene'),
                            ('disease', 'variant'),
                            ('gene', 'gene'),
                            ('gene', 'chemical'),
                            ('chemical', 'chemical'),
                            ('chemical', 'variant')
                            )

    cluster_pairs = set(ClusterPair(el1, el2) for el1, el2 in combinations(clusters, 2) if (el1.type, el2.type in valid_type_combinations))

    return cluster_pairs

def DocRED_candidate_cluster_pair_constructor(clusters):
    return set(ClusterPair(el1, el2) for el1, el2 in combinations(clusters, 2))


#%%


    
    


    
        