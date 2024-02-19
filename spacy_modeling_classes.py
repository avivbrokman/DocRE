#%% to-do list

#%% libraries
import torch
from torch.nn import LazyLinear, Embedding, Dropout
from torch.nn.functional import sigmoid, softmax
from collections import defaultdict
from itertools import combinations, product

from spacy_data_classes import SpanPair, SpanGroup, Relation, EvalMention, EvalSpanPair, EvalEntity, EvalRelation, SpanUtils
from spacy_parameter_modules import EnhancedModule

from utils import mode, unlist


#%% useful functions
def retrieve_token_embeddings_of_span(span, token_embeddings):
    return token_embeddings[span._.subword_indices[0]:span._.subword_indices[1]]

def retrieve_token_embeddings_of_path(path, token_embeddings):
    subword_indices = set()
    for el in path:
        subword_indices.update(range(*el._.subword_indices))

    subword_indices = list(subword_indices)
    return token_embeddings[subword_indices]


              


# def retrieve_token_embeddings_of_spans(spans, token_embeddings):
     
#     def workhorse(span):
#         tensors = [token_embeddings[i] for i in span._.subword_indices]
#         tensor = torch.stack(tensors)
#         return token_embeddings[:, []]

#     return [workhorse(el) for el in spans]


class SpanEmbedder(EnhancedModule):
    def __init__(self, token_embedder, token_pooler, vector_embedder, dropout_prob = 0.2):
        super().__init__()

        self.token_embedder = token_embedder
        self.token_pooler = token_pooler
        self.vector_embedder = vector_embedder
        self.dropout = Dropout(dropout_prob)
        

    def _pooled_token_embedding(self, span, token_embeddings):
        if span: # empty span
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

        # empty span
        empty_spans = list()
        pooled_embeddings = list()
        for i, el in enumerate(spans):
            # if el:
            pooled_embeddings.append(self._pooled_token_embedding(el, token_embeddings))
            if not el:
                empty_spans.append(i)
           
        pooled_embeddings = torch.stack(pooled_embeddings)

        if extra_embeddings is not None:
            pooled_embeddings = torch.cat((pooled_embeddings, extra_embeddings), dim = 1)

        pooled_embeddings = self.dropout(pooled_embeddings)
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

        span_embeddings = self.dropout(span_embeddings)

        return span_embeddings
    
class PathEmbedder(EnhancedModule):
    def __init__(self, token_embedder, token_pooler, vector_embedder, dropout_prob = 0.2):
        super().__init__()

        self.token_embedder = token_embedder
        self.token_pooler = token_pooler
        self.vector_embedder = vector_embedder
        self.dropout = Dropout(dropout_prob)
        

    def _pooled_token_embedding(self, span, token_embeddings):
        if span: # empty span
            span_token_embeddings = retrieve_token_embeddings_of_path(span, token_embeddings)
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

        # empty span
        empty_spans = list()
        pooled_embeddings = list()
        for i, el in enumerate(spans):
            # if el:
            pooled_embeddings.append(self._pooled_token_embedding(el, token_embeddings))
            if not el:
                empty_spans.append(i)
           
        pooled_embeddings = torch.stack(pooled_embeddings)

        if extra_embeddings is not None:
            pooled_embeddings = torch.cat((pooled_embeddings, extra_embeddings), dim = 1)

        pooled_embeddings = self.dropout(pooled_embeddings)
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

        span_embeddings = self.dropout(span_embeddings)

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

    # def filter_nonentities(self, spans):
    #     return set(el for el in spans if el.label_)

    def forward(self, spans, token_embeddings):
        
        length_embeddings = self.length_embedder(spans)

        span_embeddings = self.span_embedder(spans, token_embeddings, length_embeddings)

        logits = self.prediction_layer(span_embeddings)
                
        return logits#, pooled_token_embeddings

    def get_gold_labels(self, spans, entity_type_converter):
        labels = [entity_type_converter.class2index(el.label_) for el in spans]
        labels = torch.tensor(labels, device = self._device())
        return labels

    def predict(self, spans, logits, entity_type_converter):
        spans = SpanUtils.duplicate_spans(spans)

        type_indices = torch.argmax(logits, dim = 1)
        
        mentions = set()
        # eval_mentions = set()
        for i, el in enumerate(spans):
            type_ = entity_type_converter.index2class(type_indices[i])
            if type_:
                el.label_ = type_
                mentions.add(el)
                # eval_mentions.add(EvalMention.from_span(el, type_))

        return mentions#, eval_mentions
    
    def get_eval_version(self, spans):
        return {EvalMention.from_span(el) for el in spans}


#%% coreference
class Coreference(EnhancedModule):
    def __init__(self, 
                 levenshtein_embedder, levenshtein_gate, span_embedder, 
                 type_embedder, length_embedder, length_difference_embedder, 
                #  span_pair_embedder, 
                 num_coref_classes, coref_cutoff, dropout_prob = 0.2):
        super().__init__()

        self.levenshtein_embedder = levenshtein_embedder
        self.levenshtein_gate = levenshtein_gate
        self.prediction_layer = LazyLinear(num_coref_classes)

        self.span_embedder = span_embedder
        self.type_embedder = type_embedder
        self.length_embedder = length_embedder
        self.length_difference_embedder = length_difference_embedder

        self.coref_cutoff = coref_cutoff
        self.dropout = Dropout(dropout_prob)

    def get_spans_from_span_pairs(self, span_pairs):
        spans = set()
        for el in span_pairs:
            if isinstance(el, SpanPair): 
                spans.update(el.spans)
            elif isinstance(el, EvalSpanPair):
                spans.update(el.mentions)

        # spans = set.union(*[el.spans for el in span_pairs])

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

        # span_pair_embeddings = self.span_pair_embedder(span_pair_embeddings)

        span_pair_embeddings = self.dropout(span_pair_embeddings)

        logits = self.prediction_layer(span_pair_embeddings)

        return logits 

    def get_gold_labels(self, span_pairs):
        labels = [el.coref for el in span_pairs]
        labels = torch.tensor(labels, device = self._device())
        
        return labels

    def keep_coreferent_pairs(self, span_pairs):
        return [el for el in span_pairs if el.coref == 1]

    def predict(self, span_pairs, logits):

        probs = sigmoid(logits)
        # probs = softmax(logits, dim = 1) #sigmoid(logits)
        coref_indices = torch.nonzero(probs[:,1] > self.coref_cutoff).squeeze()

        eval_coreferences = set()
        for i, el in enumerate(span_pairs):
            coref = int(i in coref_indices)
            if coref:
                eval_coreferences.add(EvalSpanPair.from_span_pair(el, coref))

        return eval_coreferences
    
    def _get_objects_by_type(self, objects):
    
        object_dict = defaultdict(set)

        for el in objects:
            object_dict[el.label_].add(el)

        return object_dict

    def exhaustive_intratype_pairs(self, spans):
        spans_by_type = self._get_objects_by_type(spans)
        
        span_pairs = set()
        for class_, class_spans in spans_by_type.items():
            span_pairs.update(set(SpanPair(el1, el2) for el1, el2 in combinations(class_spans, 2)))

        return span_pairs
    
    def _get_singletons(self, mentions, coreferences):
        coreferences = set(coreferences)

        linked_mentions = self.get_spans_from_span_pairs(coreferences)  

        mentions = [EvalMention.from_span(el) for el in mentions]
        singletons = list(set(mentions) - set(linked_mentions))
 
        singletons = [EvalEntity.from_eval_mention(el) for el in singletons]

        return singletons 

    def cluster(self, mentions, coreferences):
        
        singletons = self._get_singletons(mentions, coreferences)
        unfinished_clusters = [EvalEntity.from_eval_span_pair(el) for el in coreferences]
        finished_clusters = singletons

        while unfinished_clusters:
            cluster = unfinished_clusters.pop()
            for i, el in enumerate(unfinished_clusters):
                combined = cluster.merge(el)
                if combined:
                    unfinished_clusters.pop(i)
                    unfinished_clusters.append(combined)
                    break
            else:
                finished_clusters.append(cluster)

        return finished_clusters
    
    def get_spacy_version(self, eval_entities, doc):
        return {el.to_span_group(doc) for el in eval_entities}
    
    


#%% DirectClusterer
class DirectClusterer(EnhancedModule):
    def __init__(self, 
                 levenshtein_embedder, levenshtein_gate, span_embedder, 
                 type_embedder, length_embedder, length_difference_embedder, 
                #  span_pair_embedder, 
                 num_coref_classes, coref_cutoff, dropout_prob = 0.2):
        super().__init__()

        self.levenshtein_embedder = levenshtein_embedder
        self.levenshtein_gate = levenshtein_gate
        self.prediction_layer = LazyLinear(num_coref_classes)

        self.span_embedder = span_embedder
        self.type_embedder = type_embedder
        self.length_embedder = length_embedder
        self.length_difference_embedder = length_difference_embedder

        self.coref_cutoff = coref_cutoff
        self.dropout = Dropout(dropout_prob)

    def get_spans_from_span_pairs(self, span_pairs):
        spans = set()
        for el in span_pairs:
            if isinstance(el, SpanPair): 
                spans.update(el.spans)
            elif isinstance(el, EvalSpanPair):
                spans.update(el.mentions)

        # spans = set.union(*[el.spans for el in span_pairs])

        return list(spans)

    def get_span2embedding(self, span_pairs, token_embeddings):
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

        return span2embedding

    def forward(self, span_pairs, token_embeddings):
        
        span2embedding = self.get_span2embedding(span_pairs, token_embeddings)

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

        # span_pair_embeddings = self.span_pair_embedder(span_pair_embeddings)

        span_pair_embeddings = self.dropout(span_pair_embeddings)

        logits = self.prediction_layer(span_pair_embeddings)

        return logits 

    def get_gold_labels(self, span_pairs):
        labels = [el.coref for el in span_pairs]
        labels = torch.tensor(labels, device = self._device())
        
        return labels

    def keep_coreferent_pairs(self, span_pairs):
        return [el for el in span_pairs if el.coref == 1]

    def predict(self, span_pairs, logits):

        probs = softmax(logits, dim = 1) #sigmoid(logits)
        coref_indices = torch.nonzero(probs[:,1] > self.coref_cutoff).squeeze()

        eval_span_pairs = set()
        for i, el in enumerate(span_pairs):
            coref = int(i in coref_indices)
            if coref:
                eval_span_pairs.add(EvalSpanPair.from_span_pair(el, coref))

        return eval_span_pairs
    
    def _get_objects_by_type(self, objects):
    
        object_dict = defaultdict(set)

        for el in objects:
            object_dict[el.label_].add(el)

        return object_dict

    def exhaustive_intratype_pairs(self, spans):
        spans_by_type = self._get_objects_by_type(spans)
        
        span_pairs = set()
        for class_, class_spans in spans_by_type.items():
            span_pairs.update(set(SpanPair(el1, el2) for el1, el2 in combinations(class_spans, 2)))

        return span_pairs
    
    def _get_singletons(self, mentions, coreferences):
        coreferences = set(coreferences)

        linked_mentions = self.get_spans_from_span_pairs(coreferences)  

        mentions = [EvalMention.from_span(el) for el in mentions]
        singletons = list(set(mentions) - set(linked_mentions))
 
        singletons = [EvalEntity.from_eval_mention(el) for el in singletons]

        return singletons 

    def cluster(self, mentions, coreferences):
        
        singletons = self._get_singletons(mentions, coreferences)
        unfinished_clusters = [EvalEntity.from_eval_span_pair(el) for el in coreferences]
        finished_clusters = singletons

        while unfinished_clusters:
            cluster = unfinished_clusters.pop()
            for i, el in enumerate(unfinished_clusters):
                combined = cluster.merge(el)
                if combined:
                    unfinished_clusters.pop(i)
                    unfinished_clusters.append(combined)
                    break
            else:
                finished_clusters.append(cluster)
       
        return finished_clusters



#%% RC base class
class BaseRelationClassifier(EnhancedModule):
    
    def __init__(self):
        super().__init__()
        
        self.dataset2relation_constructor = {'CDR': self.CDR_candidate_relation_constructor,
                                             'DocRED': self.DocRED_candidate_relation_constructor,
                                             'BioRED': self.BioRED_candidate_relation_constructor}

    def CDR_candidate_relation_constructor(self, entities):
    
        chemicals = set(el for el in entities if el.type == 'chemical')
        diseases = set(el for el in entities if el.type == 'disease')

        relations = set()
        relations.update(set(Relation(el1, el2) for el1, el2 in product(chemicals)))
        relations.update(set(Relation(el1, el2) for el1, el2 in product(diseases)))

        return relations

    def BioRED_candidate_relation_constructor(self, entities):
        
        valid_type_combinations = set([('DiseaseOrPhenotypicFeature',      'chemical'),
                                ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
                                ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
                                ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
                                ('GeneOrGeneProduct', 'ChemicalEntity'),
                                ('ChemicalEntity', 'ChemicalEntity'),
                                ('ChemicalEntity', 'SequenceVariant')]
                                )

        relations = set(Relation(el1, el2) for el1, el2 in combinations(entities, 2) if (el1.attrs['type'], el2.attrs['type'] in valid_type_combinations))

        return relations

    def DocRED_candidate_relation_constructor(self, entities):
        return set(Relation(el1, el2) for el1, el2 in combinations(entities, 2))
    

#%% RC
class RelationClassifier(BaseRelationClassifier):
    def __init__(self, span_embedder, intervening_span_embedder, span_pooler, type_embedder, local_pooler, relation_embedder, token_distance_embedder, sentence_distance_embedder, num_relation_classes, is_multilabel, rc_cutoff):
        
        super().__init__()

        self.span_embedder = span_embedder
        self.intervening_span_embedder = intervening_span_embedder
        self.span_pooler = span_pooler # use this or cluster_embedder
        # self.cluster_embedder = cluster_embedder # use this or span_pooler
        self.type_embedder = type_embedder
        self.local_pooler = local_pooler

        self.relation_embedder = relation_embedder
        self.token_distance_embedder = token_distance_embedder
        self.sentence_distance_embedder = sentence_distance_embedder

        self.is_multilabel = is_multilabel
        self.prediction_layer = LazyLinear(num_relation_classes)
        self.rc_cutoff = rc_cutoff

    def get_spans_from_relations(self, relations):
        spans = set()
        for relation in relations:
            spans.update(set(span for span in relation.head))
            spans.update(set(span for span in relation.tail))
        
        return list(spans)

    def get_intervening_spans(self, relations):
        spans = set()
        # if self.intervening_span_embedder:
        #     for el_cluster_pair in cluster_pairs:
        #         for el_span_pair in el_cluster_pair.enumerate_span_pairs():
        #             spans.add(el_span_pair.intervening_span())

        for relation in relations:
            for span_pair in relation.enumerate_span_pairs():
                spans.add(span_pair.intervening_span())

        return spans

    def get_entities_from_relations(self, relations):
        entities = set()
        for el in relations:
            entities.add(el.head)
            entities.add(el.tail)

        return list(entities)
    
    def embed_entity(self, entity, span2embedding):
        span_embeddings = [span2embedding[el] for el in entity]
        span_embeddings = torch.stack(span_embeddings)
        
        entity_embedding = self.span_pooler(span_embeddings)

        return entity_embedding

    def embed_entities(self, entities, span2embedding):
        entity_embeddings =  [self.embed_entity(el, span2embedding) for el in entities]
        entity_embeddings = torch.stack(entity_embeddings)
        return entity_embeddings

    def local_relation_embeddings(self, relations, span2embedding):

        def workhorse(relation):
            
            span_pairs = relation.enumerate_span_pairs()

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

        embeddings = [workhorse(el) for el in relations]
        embeddings = torch.stack(embeddings)

        return embeddings

    def global_relation_embeddings(self, relations, entity2embedding):

        def workhorse(relation):
            head_embedding = entity2embedding[relation.head]
            tail_embedding = entity2embedding[relation.tail]
            return torch.cat((head_embedding, tail_embedding), dim = 0)
            
        embeddings = [workhorse(el) for el in relations]
        return torch.stack(embeddings)
    
    def forward(self, relations, token_embeddings):
        # relations = list(relations)

        spans = self.get_spans_from_relations(relations)
        intervening_spans = self.get_intervening_spans(relations)

        # individual span embeddings
        span_embeddings = self.span_embedder(spans, token_embeddings)
        intervening_span_embeddings = self.intervening_span_embedder(intervening_spans, token_embeddings)

        span2embedding = dict(zip(spans, span_embeddings))
        intervening_span2embedding = dict(zip(intervening_spans, intervening_span_embeddings))

        # individual cluster embeddings
        entities = self.get_entities_from_relations(relations)
        
        entity_embeddings = self.embed_entities(entities, span2embedding)

        type_embeddings = self.type_embedder(entities)

        entity_embeddings = torch.cat((entity_embeddings, type_embeddings), dim = 1)

        entity2embedding = dict(zip(entities, entity_embeddings))

        # global cluster pair embeddings
        global_relation_embeddings = self.global_relation_embeddings(relations, entity2embedding)

        # local span-pair embeddings       
        local_relation_embeddings = self.local_relation_embeddings(relations, intervening_span2embedding)

        # combine global + local
        relation_embeddings = torch.cat((global_relation_embeddings, local_relation_embeddings), dim = 1)

        relation_embeddings = self.relation_embedder(relation_embeddings)

        logits = self.prediction_layer(relation_embeddings)

        return logits

    def get_gold_labels(self, relations, relation_type_converter):
        labels = [relation_type_converter.class2index(el.type) for el in relations]
        labels = torch.tensor(labels, device = self._device())
        return labels
    
    # def get_gold_labels(self, relations, pos_or_neg):
    #     val = 1 if pos_or_neg == 'pos' else 0
    #     return torch.tensor([val] * len(relations), device = self._device())
    
 
    
    def _unilabel_predict(self, relations, logits, relation_type_converter):
        type_indices = torch.argmax(logits, dim = 1)
        
        eval_relations = set()
        for i, el in enumerate(relations):
            type_ = relation_type_converter.index2class(type_indices[i])
            eval_relations.add(EvalRelation.from_relation(el, type_))
        
        return eval_relations

    def _multilabel_predict(self, relations, logits, relation_type_converter):
        probabilities = sigmoid(logits)
        predicted_indices = (probabilities > self.rc_cutoff).nonzero(as_tuple = False)
        predicted_indices = predicted_indices.tolist()

        eval_relations = set()
        for relation_index, relation_type_index in predicted_indices:
            relation = relations[relation_index]
            type_ = relation_type_converter.index2class(relation_type_index)
            eval_relations.add(EvalRelation.from_relation(relation, type_))
            

        return eval_relations

    def predict(self, relations, logits, relation_type_converter):
        if not self.is_multilabel:
            return self._unilabel_predict(relations, logits, relation_type_converter)
            
        if self.is_multilabel:
            return self._multilabel_predict(relations, logits, relation_type_converter)

    def filter_nonrelations(self, relations):
        return set(el for el in relations if el.type)


