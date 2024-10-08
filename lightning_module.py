#%% to-do list
# config parsing
# logging

#%% libraries
from os import path
import torch
from torch.nn import Embedding
from torch.nn.functional import nll_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.optim import AdamW
from lightning import LightningModule
from transformers import AutoModel
from importlib import import_module
from random import sample

from parameter_modules import EnhancedEmbedding

#%% body
class ELRELightningModule(LightningModule):
    def __init__(self, dataset_name, lm_checkpoint, ner_config, clusterer_config, rc_config, loss_coefficients, lm_learning_rate, learning_rate, ner_scorer_classes, coref_scorer_classes, cluster_scorer_classes, rc_scorer_classes, calculator_class, neg_to_pos_span_ratio, neg_to_pos_span_pair_ratio):
        super().__init__()
        self.save_hyperparameters()
        

        # gets type converters
        self.entity_type_converter = torch.load(path.join('data', 'processed', dataset_name, lm_checkpoint, 'entity_class_converter.save'))
        self.relation_type_converter = torch.load(path.join('data', 'processed', dataset_name, lm_checkpoint, 'relation_class_converter.save'))

        ner_config['config']['num_entity_classes'] = len(self.entity_type_converter)
        rc_config['config']['num_relation_classes'] = len(self.relation_type_converter) - 1

        # instantiates neural components
        self.lm = AutoModel.from_pretrained(lm_checkpoint)

        # allows BERT to handle longer sequences
        self.expand_position_embeddings(1024)

        if ner_config:
            self.ner_config = ner_config #self.recursive_instantiate(ner_config)
        if clusterer_config:
            self.clusterer_config = clusterer_config #self.recursive_instantiate(clusterer)
        if rc_config:
            self.rc_config = rc_config #self.recursive_instantiate(rc)

        if ner_config:
            self.ner = self.recursive_instantiate(ner_config)
        if clusterer_config:
            self.clusterer = self.recursive_instantiate(clusterer_config)
        if rc_config:
            self.rc = self.recursive_instantiate(rc_config)

        self.neg_to_pos_span_ratio = neg_to_pos_span_ratio
        self.neg_to_pos_span_pair_ratio = neg_to_pos_span_pair_ratio

        # loss
        self.loss_coefficients = loss_coefficients
        self.lm_learning_rate = lm_learning_rate
        self.learning_rate = learning_rate

        # gets scorer classes
        if ner_config:
            self.ner_scorer_classes = self._instantiate_scorers(ner_scorer_classes)
        
        if clusterer_config:
            self.coref_scorer_classes = self._instantiate_scorers(coref_scorer_classes)
            self.cluster_scorer_classes = self._instantiate_scorers(cluster_scorer_classes)
        if rc_config:
            self.rc_scorer_classes = self._instantiate_scorers(rc_scorer_classes)

        # instantiates calculators -- one for each scorer class
        calculators = import_module('performance_calculators')
        calculator_class = getattr(calculators, calculator_class)

        if ner_config:
            self.ner_performance_calculators = self._create_performance_calculators(self.ner_scorer_classes, calculator_class)
        if clusterer_config:
            self.coref_performance_calculators = self._create_performance_calculators(self.coref_scorer_classes, calculator_class)
            self.cluster_performance_calculators = self._create_performance_calculators(self.cluster_scorer_classes, calculator_class)
        if rc_config:
            self.rc_performance_calculators = self._create_performance_calculators(self.rc_scorer_classes, calculator_class)

        #
        self.dataset_name = dataset_name

        # creates lists for storage of details of results
        self.validation_details = list()
        self.test_details = list()

        self.validation_performance = list()
        self.test_performance = list()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        # Retrieve and use extra information for re-instantiation
        self.ner_config = checkpoint.get('ner_config')
        self.clusterer_config = checkpoint.get('clusterer_config')
        self.rc_config = checkpoint.get('rc_config')

        # Re-instantiate components if necessary
        if self.ner_config:
            self.ner = self.recursive_instantiate(self.ner_config)
        if self.clusterer_config:
            self.clusterer = self.recursive_instantiate(self.clusterer_config)
        if self.rc_config:
            self.rc = self.recursive_instantiate(self.rc_config)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # Add extra information necessary for re-instantiation
        checkpoint['ner_config'] = self.ner_config
        checkpoint['clusterer_config'] = self.clusterer_config
        checkpoint['rc_config'] = self.rc_config


    def _replace_string_with_def(self, class_info, key_name, module_names):

        if not isinstance(class_info[key_name], str):
            # If key_name is not a string, return or handle appropriately
            return

        for el in module_names:
            module = import_module(el)
            if hasattr(module, class_info[key_name]):
                class_info[key_name] = getattr(module, class_info[key_name])
                break
    
    def recursive_instantiate(self, class_info):
        print('using recursive instantiate')
        if isinstance(class_info, dict) and "class" in class_info:
            
            self._replace_string_with_def(class_info, 'class', ['modeling_classes', 'parameter_modules', 'torch.nn.functional'])

            # Check if there's a 'config' key and process it
            config = class_info.get("config", {})
            for key, value in config.items():
                config[key] = self.recursive_instantiate(value)

            # Instantiate the class with processed config
            return class_info['class'](**config)
        elif isinstance(class_info, dict) and "function" in class_info:
            # self._replace_string_with_def(class_info, 'function', ['modeling_classes', 'parameter_modules', 'torch.nn.functional'])
            # return class_info
            self._replace_string_with_def(class_info, 'function', ['modeling_classes', 'parameter_modules', 'torch.nn.functional'])
            
            return class_info['function']
        else:
            return class_info
        
    def _instantiate_scorers(self, scorer_list):
        scorers = import_module('scorers')
        return [getattr(scorers, el) for el in scorer_list]

    def _create_performance_calculators(self, scorer_classes, performance_calculator_class):
        return {el.__name__: performance_calculator_class() for el in scorer_classes}

    def expand_position_embeddings(self, new_max_length):
        original_max_length = self.lm.config.max_position_embeddings
        if original_max_length < new_max_length:
            original_embeddings = self.lm.embeddings.position_embeddings

            new_embeddings = Embedding(new_max_length, original_embeddings.weight.size(-1))
            new_embeddings.weight.data[:original_max_length, :] = original_embeddings.weight.data
            self.lm.embeddings.position_embeddings = new_embeddings
            self.lm.embeddings.register_buffer("position_ids",
                                                torch.arange(new_max_length).expand((1, -1)))

            self.lm.config.max_position_embeddings = new_max_length

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch
    
    def lm_step(self, example):
        # print('lm step')
        tokens = example.subword_tokens
        tokens = torch.tensor(tokens, dtype = torch.int32, device = self.device)
        tokens = tokens.unsqueeze(0) 
        # print('using LM')
        # self.lm.to('cpu')
        token_type_ids = torch.zeros_like(tokens, device = self.device)
        attention_mask = torch.ones_like(tokens, device = self.device)

        token_embeddings = self.lm(tokens, attention_mask, token_type_ids).last_hidden_state
        token_embeddings = token_embeddings.squeeze(0)
        # print('finished LM')
        return token_embeddings

    def subsampler(self, positives, negatives, neg_to_pos_ratio):
        num_pos = len(positives)
        num_neg = len(negatives)

        num_neg_desired = neg_to_pos_ratio * num_pos
        num_neg_desired = int(num_neg_desired)
        num_neg_desired = min(num_neg_desired, num_neg)


        negative_sample = sample(list(negatives), num_neg_desired)

        return set(positives) | set(negative_sample)

    def ner_training_step(self, example, token_embeddings):
        # print('NER training step')

        # candidate_spans = list(example.mentions | example.negative_spans)
        candidate_spans = self.subsampler(example.mentions, example.negative_spans, self.neg_to_pos_span_ratio)

        if candidate_spans:

            logits = self.ner(candidate_spans, token_embeddings)
            gold_labels = self.ner.get_gold_labels(candidate_spans, self.entity_type_converter)

            loss = cross_entropy(logits, gold_labels) #nll_loss(logits, gold_labels)
        
            return loss
        else:
            return 0



    def ner_inference_step(self, example, token_embeddings):
        # print('NER inference step')
        candidate_spans = example.candidate_spans
        
        logits = self.ner(candidate_spans, token_embeddings)
        
        predicted_spans = self.ner.predict(candidate_spans, logits, self.entity_type_converter)
        predicted_mentions = self.ner.filter_nonentities(predicted_spans)

        return predicted_mentions

    def cluster_training_step(self, example, token_embeddings):
        # print('cluster training step')
        
        # candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs

        candidate_span_pairs = self.subsampler(example.positive_span_pairs, example.negative_span_pairs, self.neg_to_pos_span_pair_ratio)
        candidate_span_pairs = list(candidate_span_pairs)


        if candidate_span_pairs:
            # print('candidate span pairs present')
            logits = self.clusterer(candidate_span_pairs, token_embeddings)
            gold_labels = self.clusterer.get_gold_labels(candidate_span_pairs)

            loss = cross_entropy(logits, gold_labels) 
            # loss = nll_loss(logits, gold_labels)

            return loss
        else: 
            print('no candidate span pairs')
            return 0

    def cluster_inference_step(self, candidate_span_pairs, mentions, token_embeddings):
        # print('cluster inference step')
        if candidate_span_pairs:
            logits = self.clusterer(candidate_span_pairs, token_embeddings)

            predicted_span_pairs = self.clusterer.predict(candidate_span_pairs, logits)
            
            predicted_coreferences = self.clusterer.keep_coreferent_pairs(predicted_span_pairs)
            
            predicted_eval_clusters = self.clusterer.cluster(mentions, predicted_coreferences)

            return predicted_coreferences, predicted_eval_clusters
        else:
            return set(), set()
    
    def rc_training_step(self, example, token_embeddings):
        
        # print('RC training step')
        # candidate_cluster_pairs = example.positive_cluster_pairs + example.negative_cluster_pairs
        
        # logits = self.rc(candidate_cluster_pairs, token_embeddings, self.rc.entity_type_converter)
        # gold_labels = self.rc.get_gold_labels(candidate_cluster_pairs)

        # loss = nll_loss(logits, gold_labels)

        if example.positive_cluster_pairs:

            positive_logits = self.rc(example.positive_cluster_pairs, token_embeddings)
            negative_logits = self.rc(example.negative_cluster_pairs, token_embeddings)

            keep_positive_logits = [positive_logits[i, self.relation_type_converter.class2index(el.type)] for i, el in enumerate(example.positive_cluster_pairs)]
            keep_negative_logits = [negative_logits[i, self.relation_type_converter.class2index(el.type)] for i, el in enumerate(example.negative_cluster_pairs)] 

            logits = keep_positive_logits + keep_negative_logits
            logits = torch.tensor(logits).to(positive_logits)


            # positive_gold_labels = self.rc.get_gold_labels(example.positive_cluster_pairs, self.relation_type_converter)
            positive_gold_labels = self.rc.get_gold_labels(example.positive_cluster_pairs, 'pos')
            # none_index = self.relation_type_converter.class2index(None)
            # negative_gold_labels = [none_index] * len(example.negative_cluster_pairs)
            # negative_gold_labels = torch.tensor(negative_gold_labels, device = self.device)
            negative_gold_labels = self.rc.get_gold_labels(example.negative_cluster_pairs, 'neg')

            gold_labels = torch.cat((positive_gold_labels, negative_gold_labels)).float()

            loss = binary_cross_entropy_with_logits(logits, gold_labels)

            return loss
        else:
            return 0

    def rc_inference_step(self, candidate_cluster_pairs, token_embeddings):
        # print('RC inference step')

        candidate_cluster_pairs = list(candidate_cluster_pairs)
        if candidate_cluster_pairs:
            logits = self.rc(candidate_cluster_pairs, token_embeddings)

            predicted_cluster_pairs = self.rc.predict(candidate_cluster_pairs, logits, self.relation_type_converter)

            predicted_relations = self.rc.filter_nonrelations(predicted_cluster_pairs)

            return predicted_relations
        else:
            return set()


    def training_step(self, example, example_index):

        # print('training step')

        token_embeddings = self.lm_step(example)

        loss = torch.tensor([0], dtype = torch.float32).to(self.device)
        if self.ner:
            ner_loss = self.ner_training_step(example, token_embeddings)
            ner_loss *= self.loss_coefficients[0]
            loss += ner_loss
            
        if self.clusterer:
            clusterer_loss = self.cluster_training_step(example, token_embeddings)
            clusterer_loss *= self.loss_coefficients[1]
            loss += clusterer_loss

        if self.rc:
            rc_loss = self.rc_training_step(example, token_embeddings)
            rc_loss *= self.loss_coefficients[2]
            loss += rc_loss

        return loss
  
    def inference_step(self, example, example_index):
        # print('inference step')
        # obtain predictions
        token_embeddings = self.lm_step(example)

        if self.ner:
            predicted_mentions = self.ner_inference_step(example, token_embeddings)
                         
        if self.clusterer:
            if self.ner:
                candidate_span_pairs = self.clusterer.exhaustive_intratype_pairs(predicted_mentions)
                predicted_coreferences, predicted_clusters = self.cluster_inference_step(candidate_span_pairs, predicted_mentions, token_embeddings)

            else:
                candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs
                predicted_coreferences, predicted_clusters = self.cluster_inference_step(candidate_span_pairs, example.mentions, token_embeddings)
            

        if self.rc:
            if self.clusterer:
                candidate_cluster_pairs = self.rc.dataset2cluster_pair_constructor[self.dataset_name](predicted_clusters)
                
            else:
                candidate_cluster_pairs = example.positive_cluster_pairs + example.negative_cluster_pairs
                
            predicted_relations = self.rc_inference_step(candidate_cluster_pairs, token_embeddings)
                
        # example performance
        # print('step performance')
        if self.ner:
            ner_scorers = dict()
            for el in self.ner_scorer_classes:
                scorer = el(predicted_mentions, example.mentions)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                ner_scorers[class_name] = scorer
                calculator = self.ner_performance_calculators[class_name]
                calculator.update(counts)
        if self.clusterer:
            coref_scorers = dict()
            for el in self.coref_scorer_classes:
                scorer = el(predicted_coreferences, example.pos_span_pairs)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                coref_scorers[class_name] = scorer
                self.coref_performance_calculators[class_name].update(**counts)
            cluster_scorers = dict()
            for el in self.cluster_scorer_classes:
                scorer = el(predicted_clusters, example.clusters)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                cluster_scorers[class_name] = scorer
                self.cluster_performance_calculators[class_name].update(counts)
        if self.rc:
            rc_scorers = dict()
            for el in self.rc_scorer_classes:
                scorer = el(predicted_relations, example.positive_cluster_pairs)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                rc_scorers[class_name] = scorer
                self.rc_performance_calculators[class_name].update(counts)

        # organizing results and details for error analysis
        details = {'example': example, 
                   'example_index': example_index
                   }
        
        if self.ner:
            details['predicted_mentions'] = predicted_mentions
            details['ner_scorers'] = ner_scorers
        if self.clusterer:
            details['predicted_coref_pairs'] = predicted_coreferences
            details['predicted_clusters'] = predicted_clusters
            details['coref_scorers'] = coref_scorers
            details['cluster_scorers'] = cluster_scorers
        if self.rc:
            details['predicted_relations'] = predicted_relations
            details['rc_scorers'] = rc_scorers
        
        return details

    def validation_step(self, example, example_index):
        if self.current_epoch > -1: #50:
            details = self.inference_step(example, example_index)
            self.validation_details[-1].append(details)


    def test_step(self, example, example_index):
        details = self.inference_step(example, example_index)
        self.test_details[-1].append(details)

    def _reset_calculators(self, calculator_dict):
        for value in calculator_dict.values():
            value.reset()

    def _compute_calculators(self, calculator_dict):
        return {key: value.compute() for key, value in calculator_dict.items()}

    def on_validation_start(self):
        # create a new list of example performance details for the new epoch
        self.validation_details.append(list())
            
    def on_validation_end(self):
        
        # obtain and save performance
        performance = dict()
        if self.ner:
            performance['ner'] = self._compute_calculators(self.ner_performance_calculators)
        if self.clusterer:
            performance['coref'] =  self._compute_calculators(self.coref_performance_calculators)
            performance['cluster'] =  self._compute_calculators(self.cluster_performance_calculators)
        if self.rc:
            performance['rc'] =  self._compute_calculators(self.rc_performance_calculators)

        self.validation_performance.append(performance)

        print(self.validation_performance)

        # resetting calculators
        if self.ner:
            self._reset_calculators(self.ner_performance_calculators)
        if self.clusterer:
            self._reset_calculators(self.coref_performance_calculators)
            self._reset_calculators(self.cluster_performance_calculators)
        if self.rc:
            self._reset_calculators(self.rc_performance_calculators)

    def configure_optimizers(self):
        print('configuring optimizers')
        lm_optimizer = {'params': self.lm.parameters(), 'lr': self.lm_learning_rate}
        ner_optimizer = {'params': self.ner.parameters(), 'lr': self.learning_rate}
        cluster_optimizer = {'params': self.clusterer.parameters(), 'lr': self.learning_rate}
        rc_optimizer = {'params': self.rc.parameters(), 'lr': self.learning_rate}
        
        return AdamW([lm_optimizer, ner_optimizer, cluster_optimizer, rc_optimizer])

    def on_train_start(self):
        self.train_loss = []

    def on_before_backward(self, loss):
        self.train_loss[-1] += loss.item()

    def on_train_epoch_start(self):
        self.train_loss.append(0)

    def on_train_epoch_end(self):
        print(self.train_loss)

    # def on_fit_end(self):
    #     torch.save(self.validation_details, path.join(self.logger.save_dir, 'validation_details.save'))
    
    # def on_validation_end(self):
    #     torch.save(self.validation_details, path.join(self.logger.save_dir, 'validation_details.save'))

    
