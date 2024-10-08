#%% to-do list
# fix up RC component

#%% libraries
import torch
from torch.nn import LazyLinear, Embedding
from torch.nn.functional import sigmoid, softmax
from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product

from data_classes import SpanPair, Cluster, ClusterPair
from parameter_modules import EnhancedModule, LazySquareLinear

from utils import generalized_replace


#%% useful functions
def retrieve_token_embeddings_of_span(span, token_embeddings):
    return token_embeddings[span.subword_indices[0]:span.subword_indices[1]]


def retrieve_token_embeddings_of_spans(spans, token_embeddings):
     
    def workhorse(span):
        tensors = [token_embeddings[i] for i in span.subword_indices]
        tensor = torch.stack(tensors)
        return token_embeddings[:, []]

    return [workhorse(el) for el in spans]






class SpanEmbedder(EnhancedModule):
    def __init__(self, token_embedder, token_pooler, vector_embedder):
        super().__init__()

        self.token_embedder = token_embedder
        self.token_pooler = token_pooler
        self.vector_embedder = vector_embedder
        

    def _pooled_token_embedding(self, span, token_embeddings):
        if span.indices: # empty span
            span_token_embeddings = retrieve_token_embeddings_of_span(span, token_embeddings)
            if span_token_embeddings.size(0) > 0:
                pooled_embedding = self.token_pooler(span_token_embeddings)
                pooled_embedding = pooled_embedding.squeeze(0)
            else:
                pass
            # empty span
            if not hasattr(self, 'pooled_embedding_size'):
                self.pooled_embedding_size = pooled_embedding.size(-1)

            return pooled_embedding
        else: 
            try:
                zero_embedding = torch.zeros(self.pooled_embedding_size, device = 'cuda')
            except:
                zero_embedding = torch.zeros(self.pooled_embedding_size)
            return zero_embedding

    def forward(self, spans, token_embeddings, extra_embeddings = None):
        token_embeddings = self.token_embedder(token_embeddings)

        # pooled_embeddings = [self._pooled_token_embedding(el, token_embeddings) for el in spans]

        # empty span
        empty_spans = list()
        pooled_embeddings = list()
        for i, el in enumerate(spans):
            # if el:
            pooled_embeddings.append(self._pooled_token_embedding(el, token_embeddings))
            if not el:
                empty_spans.append(i)
            # else:
            #     try:
            #         pooled_embeddings.append(torch.zeros(self.pooled_embedding_size, device = 'cuda'))
            #     except:
            #         pooled_embeddings.append(torch.zeros(self.pooled_embedding_size))

            #     empty_spans.append(i)

        pooled_embeddings = torch.stack(pooled_embeddings)

        if extra_embeddings is not None:
            pooled_embeddings = torch.cat((pooled_embeddings, extra_embeddings), dim = 1)

        span_embeddings = self.vector_embedder(pooled_embeddings)
        
        # empty span
        if not hasattr(self, 'null_span_embedder'):
            embedding_dim = span_embeddings.size(-1)
            try: 
                self.null_span_embedder = Embedding(1, embedding_dim, device = 'cuda')
            except:
                self.null_span_embedder = Embedding(1, embedding_dim)

        # empty span
        for i in empty_spans:
            zero_index = torch.tensor(0, device = self._device())
            span_embeddings[i] = self.null_span_embedder(zero_index)

        return span_embeddings

#%% LM
class TokenEmbedder(EnhancedModule):
    def __init__(self, dim_plm):
        super().__init__()

    def forward(self, x):
        return x



#%% NER again
class NER(EnhancedModule):
    
    def __init__(self, span_embedder, length_embedder, num_entity_classes):
        super().__init__()
        self.span_embedder = span_embedder
        self.length_embedder = length_embedder
        self.prediction_layer = LazyLinear(num_entity_classes)

    def filter_nonentities(self, spans):
        return set(el for el in spans if el.type)

    def forward(self, spans, token_embeddings):
        
        length_embeddings = self.length_embedder(spans)

        span_embeddings = self.span_embedder(spans, token_embeddings, length_embeddings)

        logits = self.prediction_layer(span_embeddings)
                
        return logits#, pooled_token_embeddings

    def get_gold_labels(self, spans, entity_type_converter):
        labels = [entity_type_converter.class2index(el.type) for el in spans]
        labels = torch.tensor(labels, device = self._device())
        return labels

    def predict(self, spans, logits, entity_type_converter):
        # entity_converter = spans[0].parent_example.parent_dataset.entity_converter
        
        type_indices = torch.argmax(logits, dim = 1)

        # spans = [generalized_replace(el, parent_sentence = None, type = entity_type_converter.index2class(type_indices[i])) for i, el in enumerate(spans)]

        spans = [el.typed_copy(entity_type_converter.index2class(type_indices[i])) for i, el in enumerate(spans)]

        return spans

#%% coreference
class Coreference(EnhancedModule):
    def __init__(self, levenshtein_embedder, levenshtein_gate, span_embedder, type_embedder, length_embedder, length_difference_embedder, num_coref_classes, coref_cutoff):
        super().__init__()

        # need
        self.levenshtein_embedder = levenshtein_embedder
        self.levenshtein_gate = levenshtein_gate
        self.prediction_layer = LazyLinear(num_coref_classes)

        self.span_embedder = span_embedder
        self.type_embedder = type_embedder
        self.length_embedder = length_embedder
        self.length_difference_embedder = length_difference_embedder

        self.coref_cutoff = coref_cutoff

    def get_spans_from_span_pairs(self, span_pairs):
        # spans = set()
        # for el in span_pairs:
        #     spans.update(el.spans)

        spans = set.union(*[el.spans for el in span_pairs])

        return list(spans)

    def forward(self, span_pairs, token_embeddings):
        
        # individual span embeddings
        spans = self.get_spans_from_span_pairs(span_pairs)

        if self.length_embedder:
            length_embeddings = self.length_embedder(spans)

        if self.length_embedder:
            span_embeddings = self.span_embedder(spans, token_embeddings, length_embeddings)
        else:
            span_embeddings = self.span_embedder(spans, token_embeddings)

        # pair embeddings
        span2embedding = dict(zip(spans, span_embeddings))

        span1_embeddings = [span2embedding[el.span1] for el in span_pairs]
        span2_embeddings = [span2embedding[el.span2] for el in span_pairs]

        span1_embeddings = torch.stack(span1_embeddings)
        span2_embeddings = torch.stack(span2_embeddings)

        span_pair_embeddings = torch.cat((span1_embeddings, span2_embeddings), dim = 1)

        # length difference embedding
        if self.length_difference_embedder:
            length_difference_embeddings = self.length_difference_embedder(span_pairs)

            span_pair_embeddings = torch.cat((span_pair_embeddings, length_difference_embeddings), dim = 1)

        if self.type_embedder:
            type_embeddings = self.type_embedder(span_pairs)

            span_pair_embeddings = torch.cat((span_pair_embeddings, type_embeddings), dim = 1)

        # levenshtein
        if self.levenshtein_embedder:
            levenshtein_embeddings = self.levenshtein_embedder(span_pairs)

            if self.levenshtein_gate: 
                levenshtein_embeddings = self.levenshtein_gate(levenshtein_embeddings, span_pair_embeddings)

            span_pair_embeddings = torch.cat((span_pair_embeddings, levenshtein_embeddings), dim = 1)

        logits = self.prediction_layer(span_pair_embeddings)

        return logits 

    def get_gold_labels(self, span_pairs):
        labels = [el.coref for el in span_pairs]
        labels = torch.tensor(labels, device = self._device())
        
        return labels

    def keep_coreferent_pairs(self, span_pairs):
        return [el for el in span_pairs if el.coref == 1]

    def predict(self, span_pairs, logits):
        
        span_pairs = generalized_replace(span_pairs, parent_example = None)
        # span_pairs = deepcopy(span_pairs)
        # for el in span_pairs:
        #     el.parent_example = None

        probs = softmax(logits, dim = 1) #sigmoid(logits)
        coref_indices = torch.nonzero(probs[:,1] > self.coref_cutoff).squeeze()

        for i, el in enumerate(span_pairs):
            el.coref = 1 if i in coref_indices else 0

        return span_pairs
    
    def _get_objects_by_type(self, objects):
    
        object_dict = defaultdict(set)

        for el in objects:
            object_dict[el.type].add(el)

        return object_dict

    def exhaustive_intratype_pairs(self, spans):
        spans_by_type = self._get_objects_by_type(spans)
        
        span_pairs = set()
        for class_, class_spans in spans_by_type.items():
            span_pairs.update(set(SpanPair(el1, el2, None, class_) for el1, el2 in combinations(class_spans, 2)))

        return span_pairs

    # def _get_singletons(self, predicted_span_pairs):
    #     coreferences = set(el for el in predicted_span_pairs if el.coref)

    #     mentions = self.get_spans_from_span_pairs(predicted_span_pairs)
    #     linked_mentions = self.get_spans_from_span_pairs(coreferences)  

    #     singletons = list(set(mentions) - set(linked_mentions))

    #     singletons = [{el} for el in singletons]

    #     return singletons 
    
    def _get_singletons(self, mentions, coreferences):
        coreferences = set(coreferences)

        linked_mentions = self.get_spans_from_span_pairs(coreferences)  

        singletons = list(set(mentions) - set(linked_mentions))

        singletons = [{el} for el in singletons]

        return singletons 


    def _combine(self, cluster1, cluster2):
        indices1 = set(el.indices for el in cluster1)
        indices2 = set(el.indices for el in cluster2)

        if indices1 & indices2:
            return cluster1 | cluster2
        
    # def cluster(self, mentions, coreferences):
        
    #     singletons = self._get_singletons(coreferences)
    #     doubletons = [el.spans for el in coreferences]
    #     unfinished_clusters = singletons + doubletons
    #     finished_clusters = list()

    #     while unfinished_clusters:
    #         cluster = unfinished_clusters.pop()
    #         for el in unfinished_clusters:
    #             combined = self._combine(cluster, el)
    #             if combined:
    #                 unfinished_clusters.append(combined)
    #                 break
    #         else:
    #             finished_clusters.append(cluster)
        
    #     clusters = set(Cluster(el) for el in finished_clusters)

    #     return clusters

    def cluster(self, mentions, coreferences):
        
        singletons = self._get_singletons(mentions, coreferences)
        doubletons = [el.spans for el in coreferences]
        unfinished_clusters = doubletons
        finished_clusters = singletons

        while unfinished_clusters:
            cluster = unfinished_clusters.pop()
            for i, el in enumerate(unfinished_clusters):
                combined = self._combine(cluster, el)
                if combined:
                    unfinished_clusters.pop(i)
                    unfinished_clusters.append(combined)
                    break
            else:
                finished_clusters.append(cluster)
        
        clusters = set(Cluster(el, None) for el in finished_clusters)

        return clusters
# #%% Iterative Clusterer
# class IterativeClusterer(nn.Module):
#     def __init__(self):

    
#     def forward(self, spans):
        



#%% RC base class
class BaseRelationClassifier(EnhancedModule):
    
    def __init__(self):
        super().__init__()
        
        self.dataset2cluster_pair_constructor = {'CDR': self.CDR_candidate_cluster_pair_constructor,
                                                 'DocRED': self.DocRED_candidate_cluster_pair_constructor,
                                                 'BioRED': self.BioRED_candidate_cluster_pair_constructor}

    def CDR_candidate_cluster_pair_constructor(self, clusters):
    
        chemicals = set(el for el in clusters if el.type == 'chemical')
        diseases = set(el for el in clusters if el.type == 'disease')

        cluster_pairs = set()
        cluster_pairs.update(set(ClusterPair(el1, el2) for el1, el2 in product(chemicals)))
        cluster_pairs.update(set(ClusterPair(el1, el2) for el1, el2 in product(diseases)))

        return cluster_pairs

    def BioRED_candidate_cluster_pair_constructor(self, clusters):
        
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

    def DocRED_candidate_cluster_pair_constructor(self, clusters):
        return set(ClusterPair(el1, el2, None) for el1, el2 in combinations(clusters, 2))
    

#%% RC
class RelationClassifier(BaseRelationClassifier):
    def __init__(self, span_embedder, intervening_span_embedder, span_pooler, type_embedder, local_pooler, cluster_pair_embedder, token_distance_embedder, sentence_distance_embedder, num_relation_classes, is_multilabel, rc_cutoff):
        
        super().__init__()

        self.span_embedder = span_embedder
        self.intervening_span_embedder = intervening_span_embedder
        self.span_pooler = span_pooler # use this or cluster_embedder
        # self.cluster_embedder = cluster_embedder # use this or span_pooler
        self.type_embedder = type_embedder
        self.local_pooler = local_pooler

        self.cluster_pair_embedder = cluster_pair_embedder
        self.token_distance_embedder = token_distance_embedder
        self.sentence_distance_embedder = sentence_distance_embedder

        self.prediction_layer = LazyLinear(num_relation_classes)
        self.is_multilabel = is_multilabel
        self.rc_cutoff = rc_cutoff


        # if self.span_pooler and self.cluster_embedder:
        #     raise ValueError('Use exactly one of span_pooler and cluster_embedder')

    def get_spans_from_cluster_pairs(self, cluster_pairs):
        spans = set()
        for el in cluster_pairs:
            spans.update(el.head.spans)
            spans.update(el.tail.spans)
        
        return list(spans)

    def get_intervening_spans(self, cluster_pairs):
        spans = set()

        for el_cluster_pair in cluster_pairs:
            for el_span_pair in el_cluster_pair.enumerate_span_pairs():
                spans.add(el_span_pair.intervening_span())

        return spans

    def get_clusters_from_cluster_pairs(self, cluster_pairs):
        clusters = set()
        for el in cluster_pairs:
            clusters.add(el.head)
            clusters.add(el.tail)

        return list(clusters)
    
    def embed_cluster(self, cluster, span2embedding):
        span_embeddings = [span2embedding[el] for el in cluster.spans]
        span_embeddings = torch.stack(span_embeddings)
        
        cluster_embedding = self.span_pooler(span_embeddings)

        return cluster_embedding

    def embed_clusters(self, clusters, span2embedding):
        cluster_embeddings =  [self.embed_cluster(el, span2embedding) for el in clusters]
        cluster_embeddings = torch.stack(cluster_embeddings)
        return cluster_embeddings

    def local_cluster_pair_embeddings(self, cluster_pairs, span2embedding):

        def workhorse(cluster_pair):
            
            span_pairs = cluster_pair.enumerate_span_pairs()

            def token_distance():

                return self.token_distance_embedder(span_pairs)

            def sentence_distance():

                return self.sentence_distance_embedder(span_pairs) 

            def intervening_span():
                
                intervening_spans = [el.intervening_span() for el in span_pairs]   
                embeddings = [span2embedding[el] for el in intervening_spans]
                embeddings = torch.stack(embeddings)

                return embeddings

            embeddings = torch.cat((token_distance(), sentence_distance(), intervening_span()), dim = 1)

            pooled_embedding = self.local_pooler(embeddings)

            return pooled_embedding

        embeddings = [workhorse(el) for el in cluster_pairs]
        embeddings = torch.stack(embeddings)

        return embeddings

    def global_cluster_pair_embeddings(self, cluster_pairs, cluster2embedding):

        def workhorse(cluster_pair):
            head_embedding = cluster2embedding[cluster_pair.head]
            tail_embedding = cluster2embedding[cluster_pair.tail]
            return torch.cat((head_embedding, tail_embedding), dim = 0)
            
        embeddings = [workhorse(el) for el in cluster_pairs]
        return torch.stack(embeddings)

    # def filter_logits(self, cluster_pairs, logits):
        
    #     types = [el.type for el in cluster_pairs]
    #     type_indices = cluster_pairs

    #     return 
    
    def forward(self, cluster_pairs, token_embeddings):
        cluster_pairs = list(cluster_pairs)

        spans = self.get_spans_from_cluster_pairs(cluster_pairs)
        intervening_spans = self.get_intervening_spans(cluster_pairs)

        # individual span embeddings
        span_embeddings = self.span_embedder(spans, token_embeddings)
        intervening_span_embeddings = self.intervening_span_embedder(intervening_spans, token_embeddings)

        span2embedding = dict(zip(spans, span_embeddings))
        intervening_span2embedding = dict(zip(intervening_spans, intervening_span_embeddings))

        # individual cluster embeddings
        clusters = self.get_clusters_from_cluster_pairs(cluster_pairs)
        
        cluster_embeddings = self.embed_clusters(clusters, span2embedding)

        type_embeddings = self.type_embedder(clusters)

        cluster_embeddings = torch.cat((cluster_embeddings, type_embeddings), dim = 1)

        cluster2embedding = dict(zip(clusters, cluster_embeddings))

        # global cluster pair embeddings
        global_cluster_pair_embeddings = self.global_cluster_pair_embeddings(cluster_pairs, cluster2embedding)

        # local span-pair embeddings       
        local_cluster_pair_embeddings = self.local_cluster_pair_embeddings(cluster_pairs, intervening_span2embedding)

        # combine global + local
        cluster_pair_embeddings = torch.cat((global_cluster_pair_embeddings, local_cluster_pair_embeddings), dim = 1)

        cluster_pair_embeddings = self.cluster_pair_embedder(cluster_pair_embeddings)

        logits = self.prediction_layer(cluster_pair_embeddings)

        return logits

    def get_gold_labels(self, cluster_pairs, relation_type_converter):
        labels = [relation_type_converter.class2index(el.type) for el in cluster_pairs]
        labels = torch.tensor(labels, device = self._device())
        return labels
    
    def get_gold_labels(self, cluster_pairs, pos_or_neg):
        val = 1 if pos_or_neg == 'pos' else 0
        return torch.tensor([val] * len(cluster_pairs), device = self._device())
        

    def _unilabel_predict(self, cluster_pairs, logits, relation_type_converter):
        type_indices = torch.argmax(logits, dim = 1)
        cluster_pairs = [generalized_replace(el, parent_example = None, type = relation_type_converter.index2class(type_indices[i])) for i, el in enumerate(cluster_pairs)]
        
        return cluster_pairs

    def _multilabel_predict(self, cluster_pairs, logits, relation_type_converter):
        probabilities = sigmoid(logits)
        predicted_indices = (probabilities > self.rc_cutoff).nonzero(as_tuple = False)
        predicted_indices = predicted_indices.tolist()

        predicted_relations = list()
        for cluster_pair_index, relation_type_index in predicted_indices:
            cluster_pair = cluster_pairs[cluster_pair_index]
            relation_type = relation_type_converter.index2class(relation_type_index)
            predicted_relation = generalized_replace(cluster_pair, parent_example = None, type = relation_type)
            predicted_relations.append(predicted_relation)

        return predicted_relations

    def predict(self, cluster_pairs, logits, relation_type_converter):
        if not self.is_multilabel:
            return self._unilabel_predict(cluster_pairs, logits, relation_type_converter)
            
        if self.is_multilabel:
            return self._multilabel_predict(cluster_pairs, logits, relation_type_converter)

    def filter_nonrelations(self, cluster_pairs):
        return set(el for el in cluster_pairs if el.type)

