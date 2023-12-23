#%% libraries
import torch
from torch.nn import Module
import os
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from utils import apply


#%% useful functions
def retrieve_token_embeddings_of_span(span, token_embeddings):
    tensors = [token_embeddings[i] for i in span.subword_indices]
    tensor = torch.stack(tensors)
    return tensor

def retrieve_token_embeddings_of_spans(spans, token_embeddings):
     
    def workhorse(span):
        tensors = [token_embeddings[i] for i in span.subword_indices]
        tensor = torch.stack(tensors)
        return tensor

    return [workhorse(el) for el in spans]




#%% token embedder
class TokenEmbedder(Module):
    
    '''Embeds ALL tokens, not just ones in some list of span.  That means that there will be some wasted embedded tokens.  But this makes code easier, and there's probably not much wasted tokens.'''
    
    def __init__(self, token_embedder):
        self.token_embedder = token_embedder

    def forward(self, token_embeddings)
        return self.token_embedder(token_embeddings)



#%% span embedder
class SpanEmbedder(Module):
    def __init__(self, token_embedder, token_pooler, span_embedder):
        self.token_embedder = token_embedder
        self.token_pooler = token_pooler
        self.span_embedder = span_embedder

    def _pooled_token_embedding(self, span, token_embeddings):
            span_token_embeddings = retrieve_token_embeddings_of_span(el)
            pooled_embedding = self.pooler(span_token_embeddings)
            return pooled_embedding

    def forward(self, spans, token_embeddings)
        token_embeddings = self.token_embedder(token_embeddings)

        pooled_embeddings = [self._pooled_token_embedding(el, token_embeddings) for el in spans]

        span_embeddings = self.span_embedder(pooled_embeddings)

        return span_embeddings


#%% LM
class TokenEmbedder(Module):
    def __init__(self, dim_plm):
        pass

    def forward(self):
        pass

#%% NER

# class NER(Module):
    
#     def __init__(self, pooler, prediction_layer, length_embedder):
#         # self.pooler = apply(pooler)
#         self.pooler = pooler
#         self.prediction_layer = prediction_layer
#         self.length_embedder = length_embedder

#     def forward(self, spans, token_embeddings):
#         span_embeddings = []
#         for el in spans:
#             token_embeddings = retrieve_token_embeddings_of_span(el)
#             pooled_token_embedding = self.pooler(token_embeddings)

#             length_embedding = self.length_embedder(len(span))
            
#             span_embedding = torch.cat((pooled_token_embedding, length_embedding))
            
#             span_embeddings.append(span_embedding)

#         span_embeddings = torch.stack(span_embeddings)
#         pooled_token_embeddings = span_embeddings[,:token_embeddings.size(2)]

#         logits = self.prediction_layer(span_embeddings)
        
#         return logits, pooled_token_embeddings

#%% NER again
class NER(Module):
    
    def __init__(self, span_embedder, length_embedder, prediction_layer):
        self.span_embedder = span_embedder
        self.length_embedder = length_embedder
        self.prediction_layer = prediction_layer

    def filter_nonentities(self, spans):
        return [el for el in spans if el.predicted_type != 'NA']

    def forward(self, spans, token_embeddings):
        
        pooled_token_embeddings = self.span_embedder(spans, token_embeddings)

        span_lengths = [len(el) for el in spans]
        length_embeddings = self.length_embedder(span_lengths)

        span_embeddings = torch.cat((pooled_token_embeddings, length_embeddings))
        
        logits = self.prediction_layer(span_embeddings)
        
        return logits#, pooled_token_embeddings

    def get_gold_labels(self, spans):
        labels = [el.type for el in spans]
        labels = torch.tensor(labels)
        
        return labels

    def predict(self, spans, logits):
        index2class = spans[0].parent_example.parent_dataset.entity_converter.index2class
        
        spans = deepcopy(spans)
        for el in spans:
            el.parent_example = None
        
        type_indices = torch.argmax(logits, dim = 1)

        for i, el in enumerate(spans):
            el.predicted_type = index2class[type_indices[i]]

        return spans

    




#%% clustering
class Coreference_with_span_embeddings(Module):
    def __init__(self, levenshtein_embedder, levenshtein_gate, prediction_layer):
        self.levenshtein_embedder = levenshtein_embedder
        self.levenshtein_gate = levenshtein_gate
        self.prediction_layer = prediction_layer

    def forward(self, span_pairs, spans, pooled_token_embeddings):
        
        positive_spans_embeddings = [(el_span, pooled_token_embeddings[i]) for i, el in enumerate(spans) if el.class != 'NA']
        span2embedding = dict(positive_spans_embeddings) 
        
        span_pair_embeddings = []
        for el in span_pairs:
            span1_pooled_token_embedding = span2embedding[el.span1]
            span2_pooled_token_embedding = span2embedding[el.span2]
            levenshtein_distance = span1.levenshtein_distance(span2)

            levenshtein_embedding = self.levenshtein_embedder[levenshtein_distance]
            gate = self.levenshtein_gate(torch.cat((span1_pooled_token_embedding, span2_pooled_token_embedding)))
            levenshtein_embedding = gate * levenshtein_embedding

            pair_embedding = torch.cat((span1_pooled_token_embedding, span2_pooled_token_embedding, levenshtein_embedding))

            span_pair_embeddings.append(pair_embedding)

        span_pair_embeddings = torch.stack(span_pair_embeddings)

        logits = self.prediction_layer(span_pair_embeddings)

        return logits
#%% non-module coreference
# class Coreference(Module):
#     def __init__(self, token_embedder, token_pooler, length_embedder, length_difference_embedder, span_embedder, levenshtein_embedder, levenshtein_gate, prediction_layer):
        
#         # optional
#         self.token_embedder = token_embedder
#         self.token_pooler = token_pooler
#         self.span_embedder = span_embedder

#         self.length_embedder = length_embedder
#         self.length_difference_embedder = length_difference_embedder

#         # need
#         self.levenshtein_embedder = levenshtein_embedder
#         self.levenshtein_gate = levenshtein_gate
#         self.prediction_layer = prediction_layer

#     def forward(self, span_pairs, spans, token_embeddings):
        
        
        

#         positive_spans_embeddings = [(el_span, pooled_token_embeddings[i]) for i, el in enumerate(spans) if el.class != 'NA']
#         span2embedding = dict(positive_spans_embeddings) 
        
#         span_pair_embeddings = []
#         for el in span_pairs:
#             span1_pooled_token_embedding = span2embedding[el.span1]
#             span2_pooled_token_embedding = span2embedding[el.span2]
#             levenshtein_distance = span1.levenshtein_distance(span2)

#             levenshtein_embedding = self.levenshtein_embedder[levenshtein_distance]
#             gate = self.levenshtein_gate(torch.cat((span1_pooled_token_embedding, span2_pooled_token_embedding)))
#             levenshtein_embedding = gate * levenshtein_embedding

#             pair_embedding = torch.cat((span1_pooled_token_embedding, span2_pooled_token_embedding, levenshtein_embedding))

#             span_pair_embeddings.append(pair_embedding)

#         span_pair_embeddings = torch.stack(span_pair_embeddings)

#         logits = self.prediction_layer(span_pair_embeddings)

#         return logits


#%% levenshtein gate
class Gate(Module):
    def __init__(self, gate):
        self.gate = gate

    def forward(span_)

#%% coreference
class Coreference(Module):
    def __init__(self, levenshtein_embedder, levenshtein_gate, prediction_layer, clusterer, span_embedder, length_embedder, length_difference_embedder,):

        # need
        self.levenshtein_embedder = levenshtein_embedder
        self.levenshtein_gate = levenshtein_gate
        self.prediction_layer = prediction_layer
        self.clusterer = clusterer

        # optional
        self.span_embedder = span_embedder
        self.length_embedder = length_embedder
        self.length_difference_embedder = length_difference_embedder

    def get_spans_from_span_pairs(self, span_pairs):
        spans = set()
        for el in span_pairs:
            spans.add(el.span1)
            spans.add(el.span2)

        return list(spans)

    def forward(self, span_pairs, token_embeddings):
        
        spans = get_spans_from_span_pairs(span_pairs)

        # individual span embeddings
        span_embeddings = self.span_embedder(spans, token_embeddings)

        if self.length_embedder:
            span_lengths = [len(el) for el in spans]
            length_embeddings = self.length_embedder(span_lengths)

            span_embeddings = torch.cat((span_embeddings, length_embeddings))
        
        # pair embeddings
        span2embedding = dict(zip(spans, span_embeddings))

        span1_embeddings = span2embedding([el.span1 for el in span_pairs])
        span2_embeddings = span2embedding([el.span2 for el in span_pairs])

        span_pair_embeddings = torch.cat(span1_embeddings, span2_embeddings)

        if self.levenshtein_embedder:
            levenshtein_distances = [el.levenshtein_distance() for el in span_pairs]
            levenshtein_embedding = self.levenshtein_embedder(levenshtein_distance)


        if self.levenshtein_gate:
            
            gate = self.levenshtein_gate(torch.cat((span1_pooled_token_embedding, span2_pooled_token_embedding)))
            levenshtein_embedding = gate * levenshtein_embedding

        if self.levenshtein_embedder:
            span_pair_embeddings = torch.cat((span_pair_embeddings, levenshtein_embeddings))

        if self.length_difference_embedder:
            length_differences = [el.length_difference() for el in span_pairs]
            length_difference_embeddings = self.length_difference_embedder(length_differences)

            span_pair_embeddings = torch.cat((span_pair_embeddings, length_difference_embedder))

        logits = self.prediction_layer(span_pair_embeddings)

        return logits 

    def get_gold_labels(self, span_pairs):
        labels = [el.coref for el in spans_pairs]
        labels = torch.tensor(labels)
        
        return labels

    def keep_coreferent_pairs(self, span_pairs):
        return [el for el in span_pairs if el.predicted_coref == 1]

    def predict(self, span_pairs, logits):
        
        span_pairs = deepcopy(span_pairs)
        for el in spans:
            el.parent_example = None

        coref_indices = torch.argmax(logits, dim = 1)

        for i, el in enumerate(span_pairs):
            el.predicted_coref = type_indices[i]

        return span_pairs
        



#%% RC
class RelationClassifier(Module):
    def __init__(self, local_pooler, prediction_layer,span_embedder, intervening_span_embedder, span_pooler, cluster_embedder, type_embedder):
    
        # necessary
        self.local_pooler = local_pooler
        self.prediction_layer = prediction_layer
        
        # optional
        self.span_embedder = span_embedder
        self.intervening_span_embedder = intervening_span_embedder
        self.span_pooler = span_pooler # use this or cluster_embedder
        self.cluster_embedder = cluster_embedder # use this or span_pooler
        self.type_embedder = type_embedder

        if self.span_pooler and self.cluster_embedder:
            raise ValueError('Use exactly one of span_pooler and cluster_embedder')

    def get_spans_from_cluster_pairs(self):
        spans = set()
        for el in cluster_pairs:
            spans.update(el.head.spans)
            spans.update(el.tail.spans)
        
        return list(spans)

    def get_intervening_spans(self):
        spans = set()
        if self.intervening_span_embedder:
            for el in cluster_pairs:
                for el_pair in el.enumerate_span_pairs():
                    spans.add(el_pair.intervening_span())

        return spans

    def get_clusters_from_cluster_pairs(self):
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
        return [self.embed_cluster(el, span2embedding) for el in clusters]

    def local_cluster_pair_embeddings(self, cluster_pairs, span2embedding):

        def workhorse(cluster_pair):
            
            span_pairs = cluster_pair.enumerate_span_pairs()

            def token_distance():

                distances = [el.token_distance() for el in span_pairs]

                embeddings = self.token_distance_embedder(distances)

                return embeddings

            def sentence_distance():

                distances = [el.sentence_distance() for el in span_pairs]

                embeddings = self.sentence_distance_embedder(distances) 

                return embeddings

            def intervening_span():
                
                intervening_spans = [el.intervening_span() for el in span_pairs]   
                embeddings = [span2embedding[el] for el in intervening_spans]
                embeddings = torch.stack(embeddings)

                return embeddings

            embeddings = torch.cat((token_distance(), sentence_distance(), intervening_span()))

            pooled_embedding = self.local_pooler(embeddings)

            return pooled_embedding

        embeddings = [workhorse(el) for el in cluster_pairs]
        embeddings = torch.cat(embeddings)

        return embeddings

    def global_cluster_pair_embeddings(self, cluster_pairs, cluster2embedding):

        def workhorse(cluster_pair):
            head_embedding = cluster2embedding[cluster_pair.head]
            tail_embedding = cluster2embedding[cluster_pair.tail]
            return torch.cat((head_embedding, tail_embedding))
            
        embeddings = [workhorse(el for el in cluster_pairs)]
        return torch.stack(embeddings)

    def forward(self, cluster_pairs, token_embeddings, entity_type_converter):

        spans = self.get_spans_from_cluster_pairs()
        intervening_spans = self.get_intervening_spans()

        # individual span embeddings
        span_embeddings = self.span_embedder(spans, token_embeddings)
        intervening_span_embeddings = self.intervening_span_embedder(intervening_spans, intervening_span_embeddings)

        span2embedding = dict(zip(spans, span_embeddings))
        intervening_span2embedding = dict(zip(intervening, intervening_span_embeddings))

        # individual cluster embeddings
        cluster_embeddings = self.embed_clusters(el, span2embedding)

        types = [entity_type_converter.class2index(el.type) for el in clusters]
        type_embeddings = self.type_embedder(types)

        cluster_embeddings = torch.cat((cluster_embeddings, type_embeddings))

        cluster2embedding = dict(zip(clusters, cluster_embeddings))

        # global cluster pair embeddings
        global_cluster_pair_embeddings = self.global_cluster_pair_embeddings(cluster_pairs, cluster2embedding)

        # local span-pair embeddings       
        local_cluster_pair_embeddings = self.local_cluster_pair_embeddings(cluster_pairs, intervening_span2embedding)

        # combine global + local
        cluster_pair_embeddings = torch.cat((global_cluster_pair_embeddings, local_cluster_pair_embeddings))

        logits = self.prediction_layer(cluster_pair_embeddings)

        return logits

    def get_gold_labels(self, cluster_pairs):
        return [el.type for el in cluster_pairs]

    def predict(self, cluster_pairs, logits, entity_type_converter):
        
        cluster_pairs = deepcopy(cluster_pairs)
        for el in cluster_pairs:
            el.parent_example = None

        type_indices = torch.argmax(logits, dim = 1)

        for i, el in enumerate(cluster_pairs):
            el.predicted_type = entity_type_converter.index2class[type_indices[i]]

        return cluster_pairs

    def filter_nonrelations(self, cluster_pairs):
        return [el for el in cluster_pairs if el.type != 'NA']


    


        
            

            
        
            


    

