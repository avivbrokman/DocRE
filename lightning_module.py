#%% libraries
import torch
from torch.nn.functional import nll_loss
from lightning import LightningModule

#%% body
class LightningRE(LightningModule):
    def __init__(self, lm, ner, clusterer, entity_type_converter, relation_type_converter, loss_coefficients, ner_scorer_classes, coref_scorer_classes, cluster_scorer_classes, rc_scorer_classes):
        super().__init__()
        self.lm = lm
        self.ner = ner
        self.clusterer = clusterer
        self.rc = rc

        self.entity_type_converter = entity_type_converter
        self.relation_type_converter = relation_type_converter

        self.loss_coefficients = loss_coefficients

        self.ner_scorer_classes = ner_scorer_classes
        self.coref_scorer_classes = coref_scorer_classes
        self.cluster_scorer_classes = cluster_scorer_classes
        self.rc_scorer_classes = rc_scorer_classes

        self.validation_details = []

    def lm_step(self, example):
        tokens = example.subword_tokens
        token_embeddings = self.lm(tokens)

        return token_embeddings

    def ner_training_step(self, example, token_embeddings):
        candidate_spans = example.mentions + example.negative_spans
        
        logits = self.ner(candidate_spans, token_embeddings)
        gold_labels = self.ner.get_gold_labels(candidate_spans)

        loss = nll_loss(logits, gold_labels)
    
        return loss

    def ner_inference_step(self, example, token_embeddings):
        candidate_spans = example.candidate_spans
        
        logits = self.ner(candidate_spans, token_embeddings)
        
        predicted_spans = self.ner.predict(candidate_spans, logits)
        predicted_mentions = self.ner.filter_nonentities(predicted_spans)

        return predicted_mentions

    def cluster_training_step(self, example, token_embeddings):
        candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs
        
        logits = self.clusterer(candidate_span_pairs, token_embeddings)
        gold_labels = self.clusterer.get_gold_labels(candidate_span_pairs)

        loss = nll_loss(logits, gold_labels)

        return loss

    def cluster_inference_step(self, example, token_embeddings):
        candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs
        
        logits = self.clusterer(candidate_span_pairs, token_embeddings)

        predicted_span_pairs = self.clusterer.predict(candidate_span_pairs, logits)
        
        predicted_coreferent_pairs = self.clusterer.keep_coreferent_pairs(predicted_span_pairs)
        
        predicted_clusters = self.clusterer.clusterer(predicted_span_pairs)

        return predicted_coreferent_pairs, predicted_clusters    
    
    def rc_training_step(self, example, entity_type_converter):
        candidate_cluster_pairs = example.positive_cluster_pairs + example.negative_cluster_pairs
        
        logits = self.rc(candidate_cluster_pairs, token_embeddings, entity_type_converter)
        gold_labels = self.rc.get_gold_labels(candidate_cluster_pairs)

        loss = nll_loss(logits, gold_labels)

        return loss

    def rc_inference_step(self, example, entity_type_converter):
        candidate_cluster_pairs = example.candidate_cluster_pairs

        logits = self.rc(candidate_cluster_pairs, token_embeddings, entity_type_converter)

        predicted_cluster_pairs = self.rc.predict(candidate_cluster_pairs, logits, entity_type_converter)

        predicted_relations = self.rc.filter_nonrelations(predicted_cluster_pairs)

        return predicted_relations


    def training_step(self, example, example_index):

        token_embeddings = self.lm_step(example)

        loss = torch.tensor([0])
        if self.ner:
            ner_loss = self.ner_training_step(example, token_embeddings)
            ner_loss *= loss_coefficients['ner']
            loss += ner_loss
            
        if self.clusterer:
            clusterer_loss = self.cluster_training_step(example, token_embeddings)
            clusterer_loss *= loss_coefficients['cluster']
            loss += clusterer_loss

        if self.rc:
            rc_loss = self.rc_training_step(example, token_embeddings)
            rc_loss *= loss_coefficients['rc']
            loss += clusterer_loss

        return loss
  s
    def inference_step(self, example, example_index):
        
        token_embeddings = self.lm_step(example)

        if self.ner:
            predicted_mentions = self.ner_inference_step(example, token_embeddings)
                         
        if self.clusterer:
            if self.ner:
                # NEED TO ALTER CODE SO THAT cluster_inference_step() WILL ACCEPT predicteD_mentions
            else:
                predicted_coreferent_pairs, predicted_clusters = self.cluster_inference_step(example, token_embeddings)
            
            coref_scorers = dict()
                for el in self.coref_scorer_classes:
                    class_name = type(el).__name__
                    coref_scorers[class_name] = el(predicted_coreferent_pairs, example.pos_span_pairs)

            cluster_scorers = dict()
            for el in self.cluster_scorer_classes:
                class_name = type(el).__name__
                cluster_scorers[class_name] = el(predicted_mentions, example.positive_spans)

        if self.rc:
            if self.cluster:
                # NEED TO ALTER CODE SO THAT rc_inference_step() WILL ACCEPT predicted_clusters
            else:
                predicted_relations = self.rc_inference_step(example, token_embeddings)
        
                        cluster_scorers = dict()
            
            rc_scorers = set()
            for el in self.rc_scorer_classes:
                class_name = type(el).__name__
                rc_scorers[class_name] = el(predicted_relations, example.positive_cluster_pairs)


        if self.ner:
            ner_scorers = dict()
            for el in self.ner_scorer_classes:
                class_name = type(el).__name__
                ner_scorers[class_name] = el(predicted_mentions, example.positive_spans)

        details = {'example': example, 
                    'example_index': example_index,
                    'predicted_mentions': predicted_mentions,
                    'predicted_coref_pairs': predicted_coreferent_pairs,
                    'predicted_clusters': predicted_clusters,
                    'predicted_relations': predicted_relations
                    }
        

        validation_details.append({'example': exa})
        
        return 





    def on_validation_epoch_end(self)

    
    def test_step(self):

    def configure_optimizers(self):

    