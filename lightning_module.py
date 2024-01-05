#%% to-do list
# config parsing
# logging

#%% libraries
from os import path
import torch
from torch.nn.functional import nll_loss
from torch.optim import AdamW
from lightning import LightningModule
from transformers import AutoModel
from importlib import import_module

#%% body
class ELRELightningModule(LightningModule):
    def __init__(self, dataset_name, lm_checkpoint, ner, clusterer, rc, loss_coefficients, lm_learning_rate, learning_rate, ner_scorer_classes, coref_scorer_classes, cluster_scorer_classes, rc_scorer_classes, calculator_class):
        super().__init__()
        self.save_hyperparameters()
        

        # gets type converters
        self.entity_type_converter = torch.load(path.join('data', 'processed', dataset_name, 'entity_type_converter.save'))
        self.relation_type_converter = torch.load(path.join('data', 'processed', dataset_name, 'relation_type_converter.save'))

        ner['num_entity_classes'] = len(self.entity_type_converter)
        rc['num_relation_classes'] = len(self.relation_type_converter)

        # instantiates neural components
        self.lm = AutoModel.from_pretrained(lm_checkpoint)

        # allows BERT to handle longer sequences
        max_sequence_length = 1024
        self.lm.resize_token_embeddings(max_sequence_length)
        self.lm.config.max_position_embeddings = max_sequence_length

        if ner:
            self.ner = ner
        if clusterer:
            self.clusterer = clusterer
        if rc:
            self.rc = rc

        # loss
        self.loss_coefficients = loss_coefficients
        self.lm_learning_rate = lm_learning_rate
        self.learning_rate = learning_rate

        # gets scorer classes
        if ner:
            self.ner_scorer_classes = ner_scorer_classes
        
        if clusterer:
            self.coref_scorer_classes = coref_scorer_classes
            self.cluster_scorer_classes = cluster_scorer_classes
        if rc:
            self.rc_scorer_classes = rc_scorer_classes

        # instantiates calculators -- one for each scorer class
        if ner:
            self.ner_performance_calculators = self._create_performance_calculators(ner_scorer_classes, calculator_class)
        if clusterer:
            self.coref_performance_calculators = self._create_performance_calculators(coref_scorer_classes, calculator_class)
            self.cluster_performance_calculators = self._create_performance_calculators(cluster_scorer_classes, calculator_class)
        if rc:
            self.rc_performance_calculators = self._create_performance_calculators(rc_scorer_classes, calculator_class)

        # creates lists for storage of details of results
        self.validation_details = list()
        self.test_details = list()

        self.validation_performance = list()
        self.test_performance = list()

    # def setup(self, stage = None):
    #     self.entity_type_converter = getattr(self.trainer.datamodule, 'entity_type_converter', None)
    #     self.relation_type_converter = getattr(self.trainer.datamodule, 'relation_type_converter', None)

    import importlib

    def recursive_instantiate(self, config):
        if isinstance(config, dict):
            if 'function' in config:
                try:
                    module = import_module('parameter_modules')
                    config['function'] = getattr(module, config['function'])
                except:
                    module = import_module('modeling_classes')
                    config['function'] = getattr(module, config['function'])
            if 'class' in config:
                class_name = config['class']
                class_config = config.get('config', {})
                # module_name, class_name = class_name.rsplit('.', 1)
                try:
                    module = import_module('modeling_classes')
                    cls = getattr(module, class_name)
                except:
                    module = import_module('parameter_modules')
                    cls = getattr(module, class_name)
                return cls(**self.recursive_instantiate(class_config))

            else:
                return {k: self.recursive_instantiate(v) for k, v in config.items()}
        return config


    def _create_performance_calculators(self, scorer_classes, performance_calculator_class):
        return {type(el).__name__: performance_calculator_class for el in scorer_classes}

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

    def cluster_inference_step(self, candidate_span_pairs, token_embeddings):
        
        logits = self.clusterer(candidate_span_pairs, token_embeddings)

        predicted_span_pairs = self.clusterer.predict(candidate_span_pairs, logits)
        predicted_clusters = self.clusterer.cluster(predicted_span_pairs)
        
        predicted_coreferent_pairs = self.clusterer.keep_coreferent_pairs(predicted_span_pairs)
        
        predicted_clusters = self.clusterer.clusterer(predicted_span_pairs)

        return predicted_coreferent_pairs, predicted_clusters    
    
    def rc_training_step(self, example, token_embeddings):
        candidate_cluster_pairs = example.positive_cluster_pairs + example.negative_cluster_pairs
        
        logits = self.rc(candidate_cluster_pairs, token_embeddings, self.trainer.datamodule.entity_type_converter)
        gold_labels = self.rc.get_gold_labels(candidate_cluster_pairs)

        loss = nll_loss(logits, gold_labels)

        return loss

    def rc_inference_step(self, candidate_cluster_pairs, token_embeddings):

        logits = self.rc(candidate_cluster_pairs, token_embeddings, self.trainer.datamodule.entity_type_converter)

        predicted_cluster_pairs = self.rc.predict(candidate_cluster_pairs, logits, self.trainer.datamodule.entity_type_converter)

        predicted_relations = self.rc.filter_nonrelations(predicted_cluster_pairs)

        return predicted_relations


    def training_step(self, example, example_index):

        token_embeddings = self.lm_step(example)

        loss = torch.tensor([0])
        if self.ner:
            ner_loss = self.ner_training_step(example, token_embeddings)
            ner_loss *= self.loss_coefficients['ner']
            loss += ner_loss
            
        if self.clusterer:
            clusterer_loss = self.cluster_training_step(example, token_embeddings)
            clusterer_loss *= self.loss_coefficients['cluster']
            loss += clusterer_loss

        if self.rc:
            rc_loss = self.rc_training_step(example, token_embeddings)
            rc_loss *= self.loss_coefficients['rc']
            loss += clusterer_loss

        return loss
  
    def inference_step(self, example, example_index):
        
        # obtain predictions
        token_embeddings = self.lm_step(example)

        if self.ner:
            predicted_mentions = self.ner_inference_step(example, token_embeddings)
                         
        if self.clusterer:
            if self.ner:
                candidate_span_pairs = self.clusterer.exhaustive_intratype_pairs(predicted_mentions)
            else:
                candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs

            predicted_coreferent_pairs, predicted_clusters = self.cluster_inference_step(candidate_span_pairs, token_embeddings)

        if self.rc:
            if self.cluster:
                candidate_cluster_pairs = self.rc.self.rc.dataset2cluster_pair_constructor[self.dataset_name](predicted_clusters)
                
            else:
                candidate_cluster_pairs = example.positive_cluster_pairs + example.negative_cluster_pairs
                
            predicted_relations = self.rc_inference_step(candidate_cluster_pairs, token_embeddings)
                
        # example performance
        if self.ner:
            ner_scorers = dict()
            for el in self.ner_scorer_classes:
                class_name = type(el).__name__
                ner_scorers[class_name] = el(predicted_mentions, example.positive_spans)
                self.ner_performance_calculators[class_name].update(**el.performants_counts())
        if self.clusterer:
            coref_scorers = dict()
            for el in self.coref_scorer_classes:
                class_name = type(el).__name__
                coref_scorers[class_name] = el(predicted_coreferent_pairs, example.pos_span_pairs)
                self.coref_performance_calculators[class_name].update(**el.performants_counts())
            cluster_scorers = dict()
            for el in self.cluster_scorer_classes:
                class_name = type(el).__name__
                cluster_scorers[class_name] = el(predicted_clusters, example.clusters)
                self.cluster_performance_calculators[class_name].update(**el.performants_counts())
        if self.rc:
            rc_scorers = set()
            for el in self.rc_scorer_classes:
                class_name = type(el).__name__
                rc_scorers[class_name] = el(predicted_relations, example.positive_cluster_pairs)
                self.rc_performance_calculators[class_name].update(**el.performants_counts())

        # organizing results and details for error analysis
        details = {'example': example, 
                   'example_index': example_index
                   }
        
        if self.ner:
            details['predicted_mentions'] = predicted_mentions
            details['ner_scorers'] = ner_scorers
        if self.cluster:
            details['predicted_coref_pairs'] = predicted_coreferent_pairs
            details['predicted_clusters'] = predicted_clusters
            details['coref_scorers'] = coref_scorers
            details['cluster_scorers'] = cluster_scorers
        if self.rc:
            details['predicted_relations'] = predicted_relations
            details['rc_scorers'] = rc_scorers
        
        return details

    def validation_step(self, example, example_index):
        details = self.inference_step(example, example_index)
        self.validation_details[-1].append(details)


    def test_step(self, example, example_index):
        details = self.inference_step(example, example_index)
        self.test_details[-1].append(details)

    def _reset_calculators(self, calculator_dict):
        for value in calculator_dict.values():
            value.reset()

    def _compute_calculators(self, calculator_dict):
        return {key: value.compute() for key, value in self.calculator_dict.items()}

    def on_validation_epoch_start(self):
        # create a new list of example performance details for the new epoch
        self.details.append(list())
            
    def on_validation_epoch_end(self):
        
        # obtain and save performance
        performance = dict()
        if self.ner:
            performance['ner'] = self._compute_calculators(self.ner_performance_calculators.items())
        if self.cluster:
            performance['coref'] =  self._compute_calculators(self.coref_performance_calculators.items())
            performance['cluster'] =  self._compute_calculators({key: value.compute() for key, value in self.cluster_performance_calculators.items()})
        if self.rc:
            performance['rc'] =  self._compute_calculators(self.rc_performance_calculators.items())

        self.validation_performance.append(performance)

        # resetting calculators
        if self.ner:
            self._reset_calculators(self.ner_performance_calculators)
        if self.cluster:
            self._reset_calculators(self.coref_performance_calculators)
            self._reset_calculators(self.cluster_performance_calculators)
        if self.rc:
            self._reset_calculators(self.rc_performance_calculators)

    def configure_optimizers(self):
        lm_optimizer = {'params': self.lm.parameters(), 'lr': self.lm_learning_rate}
        ner_optimizer = {'params': self.ner.parameters(), 'lr': self.learning_rate}
        cluster_optimizer = {'params': self.cluster.parameters(), 'lr': self.learning_rate}
        rc_optimizer = {'params': self.rc.parameters(), 'lr': self.learning_rate}
        
        return AdamW([lm_optimizer, ner_optimizer, cluster_optimizer, rc_optimizer])
        
