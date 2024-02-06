#%% to-do list
# Add logic for producing Eval versions of predictions for scoring

#%% libraries
import os
from os import path
import re
import torch
from torch.nn import Embedding
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.optim import AdamW
from lightning import LightningModule
from transformers import AutoModel
from importlib import import_module
from random import sample

# from parameter_modules import EnhancedEmbedding
from spacy_data_classes import SpanUtils
from utils import save_json

#%% body
class ELRELightningModule(LightningModule):
    def __init__(self, 
                 dataset_name, 
                 lm_checkpoint, 
                 task,
                 ner_config, clusterer_config, rc_config, 
                 loss_coefficients, 
                 lm_learning_rate, learning_rate, 
                 ner_scorer_classes, coref_scorer_classes, entity_scorer_classes, rc_scorer_classes, 
                 calculator_class, 
                 neg_to_pos_span_ratio, neg_to_pos_span_pair_ratio, neg_to_pos_relation_ratio,
                 span_sampling_probs, 
                 max_span_length):
        super().__init__()
        self.save_hyperparameters()
        
        self.task = task

        # gets type converters
        self.entity_type_converter = torch.load(path.join('data', 'processed', dataset_name, lm_checkpoint, 'entity_class_converter.save'))
        
        # if self.task in ['rc', 'e2e']:
        #     self.relation_type_converter = torch.load(path.join('data', 'processed', dataset_name, lm_checkpoint, 'relation_class_converter.save'))

        self.relation_type_converter = torch.load(path.join('data', 'processed', dataset_name, lm_checkpoint, 'relation_class_converter.save'))

        # dynamically add number of classes to modeling configs so the user doesn't need to specify them
        if self.task in ['ner', 'e2e']:
            ner_config['config']['num_entity_classes'] = len(self.entity_type_converter)
        if self.task in ['rc', 'e2e']:
            rc_config['config']['num_relation_classes'] = len(self.relation_type_converter)
           
            
            # if the task is multilabel, there should be no null class classifier, so remove one from the number of relation classes
            if rc_config['config']['is_multilabel']:
                rc_config['config']['num_relation_classes'] -= 1

        # instantiates neural components
        self.lm = AutoModel.from_pretrained(lm_checkpoint)

        # allows language model to handle longer sequences
        self.expand_position_embeddings(1024)

        # keep configs
        if self.task in ['ner', 'e2e']:
            self.ner_config = ner_config 
        if self.task in ['cluster', 'e2e']:
            self.clusterer_config = clusterer_config
        if self.task in ['rc', 'e2e']:
            self.rc_config = rc_config

        # instantiating models
        if self.task in ['ner', 'e2e']:
            self.ner = self.recursive_instantiate(ner_config)
        if self.task in ['cluster', 'e2e']:
            self.clusterer = self.recursive_instantiate(clusterer_config)
        if self.task in ['rc', 'e2e']:
            self.rc = self.recursive_instantiate(rc_config)

        # settings for sampling negative examples
        if self.task in ['ner', 'e2e']:
            self.neg_to_pos_span_ratio = neg_to_pos_span_ratio
            self.multi_shift_prob = span_sampling_probs['multi_shift']
            self.short_prob = span_sampling_probs['short']
        if self.task in ['cluster', 'e2e']:
            self.neg_to_pos_span_pair_ratio = neg_to_pos_span_pair_ratio
        if self.task in ['rc', 'e2e']:
            self.neg_to_pos_relation_ratio = neg_to_pos_relation_ratio

        # loss
        if self.task == 'ner':
            self.loss_coefficients = [1,0,0]
        if self.task == 'cluster': 
            self.loss_coefficients = [0,1,0]
        if self.task == 'rc':
            self.loss_coefficients = [0,0,1]
        if self.task == 'e2e':
            self.loss_coefficients = loss_coefficients
            
        # learning rate
        self.lm_learning_rate = lm_learning_rate
        self.learning_rate = learning_rate

        # gets scorer classes
        if self.task in ['ner', 'e2e']:
            self.ner_scorer_classes = self._instantiate_scorers(ner_scorer_classes)
        
        if self.task in ['cluster', 'e2e']:
            self.coref_scorer_classes = self._instantiate_scorers(coref_scorer_classes)
            self.entity_scorer_classes = self._instantiate_scorers(entity_scorer_classes)
        if self.task in ['rc', 'e2e']:
            self.rc_scorer_classes = self._instantiate_scorers(rc_scorer_classes)

        # instantiates calculators -- one for each scorer class
        calculators = import_module('performance_calculators')
        calculator_class = getattr(calculators, calculator_class)

        if self.task in ['ner', 'e2e']:
            self.ner_performance_calculators = self._create_performance_calculators(self.ner_scorer_classes, calculator_class)
        if self.task in ['cluster', 'e2e']:
            self.coref_performance_calculators = self._create_performance_calculators(self.coref_scorer_classes, calculator_class)
            self.entity_performance_calculators = self._create_performance_calculators(self.entity_scorer_classes, calculator_class)
        if self.task in ['rc', 'e2e']:
            self.rc_performance_calculators = self._create_performance_calculators(self.rc_scorer_classes, calculator_class)

        # sets dataset_name
        self.dataset_name = dataset_name
        
        # sets evaluation maximum candidate span length
        if self.task in ['ner', 'e2e']:
            self.max_span_length = max_span_length

        # creates lists for storage of details of results
        # self.validation_details = list()
        # self.test_details = list()

        # self.validation_performance = list()
        # self.test_performance = list()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        # Retrieve and use extra information for re-instantiation
        if self.task in ['ner', 'e2e']:
            self.ner_config = checkpoint.get('ner_config')
            self.ner = self.recursive_instantiate(self.ner_config)
        if self.task in ['cluster', 'e2e']:
            self.clusterer_config = checkpoint.get('clusterer_config')
            self.clusterer = self.recursive_instantiate(self.clusterer_config)
        if self.task in ['rc', 'e2e']:
            self.rc_config = checkpoint.get('rc_config')
            self.rc = self.recursive_instantiate(self.rc_config)


        # # Re-instantiate components if necessary
        # if self.ner_config:
        #     self.ner = self.recursive_instantiate(self.ner_config)
        # if self.clusterer_config:
        #     self.clusterer = self.recursive_instantiate(self.clusterer_config)
        # if self.rc_config:
        #     self.rc = self.recursive_instantiate(self.rc_config)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # Add extra information necessary for re-instantiation
        if self.task in ['ner', 'e2e']:
            checkpoint['ner_config'] = self.ner_config
        if self.task in ['cluster', 'e2e']:
            checkpoint['clusterer_config'] = self.clusterer_config
        if self.task in ['rc', 'e2e']:
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
            
            self._replace_string_with_def(class_info, 'class', ['spacy_modeling_classes', 'spacy_parameter_modules', 'torch.nn.functional'])

            # Check if there's a 'config' key and process it
            config = class_info.get("config", {})
            for key, value in config.items():
                config[key] = self.recursive_instantiate(value)

            # Instantiate the class with processed config
            return class_info['class'](**config)
        elif isinstance(class_info, dict) and "function" in class_info:
            # self._replace_string_with_def(class_info, 'function', ['modeling_classes', 'parameter_modules', 'torch.nn.functional'])
            # return class_info
            self._replace_string_with_def(class_info, 'function', ['spacy_modeling_classes', 'spacy_parameter_modules', 'torch.nn.functional'])
            
            return class_info['function']
        else:
            return class_info
        
    def _instantiate_scorers(self, scorer_list):
        scorers = import_module('spacy_scorers')
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

    def coref_subsampler(self, positives, negatives, neg_to_pos_ratio):
                
        num_pos = len(positives)
        num_neg = len(negatives)

        num_neg_desired = neg_to_pos_ratio * num_pos
        num_neg_desired = int(num_neg_desired)
        num_neg_desired = min(num_neg_desired, num_neg)

        negative_sample = sample(list(negatives), num_neg_desired)

        return set(positives) | set(negative_sample)

    def ner_subsampler(self, positives, negatives):
        
        num_pos = len(positives)
        num_neg = sum([len(el) for el in negatives])

        num_neg_desired = self.neg_to_pos_span_ratio * num_pos
        num_neg_desired = int(num_neg_desired)
        num_neg_desired = min(num_neg_desired, num_neg)

        one_shift_spans, multi_shift_subspans, short_spans = negatives

        num_leftover_neg_desired = max((num_neg_desired - len(one_shift_spans), 0))
        num_multi_shift_desired = int(self.multi_shift_prob * num_leftover_neg_desired)
        num_short_desired = int(self.short_prob * num_leftover_neg_desired)

        num_one_shift_available = len(one_shift_spans)    
        num_multi_shift_available = len(multi_shift_subspans)
        num_short_available = len(short_spans)
        
        if num_one_shift_available > num_neg_desired:
            one_shift_sample = sample(one_shift_spans, num_neg_desired)
        else:
            one_shift_sample = one_shift_spans

        if num_multi_shift_available > num_multi_shift_desired:
            multi_shift_sample = sample(list(multi_shift_subspans), num_multi_shift_desired)
        elif num_multi_shift_available == num_multi_shift_desired:
            multi_shift_sample = multi_shift_subspans
        elif num_multi_shift_available < num_multi_shift_desired:
            multi_shift_sample = multi_shift_subspans
            num_short_desired += (num_multi_shift_desired - num_multi_shift_available)
        
        short_sample = sample(list(short_spans), num_short_desired)

        return set(positives) | set(one_shift_sample) | set(multi_shift_sample) | set(short_sample)

    def rc_subsampler(self, positives, negatives):
        num_pos = len(positives)
        num_neg = len(negatives)

        num_neg_desired = self.neg_to_pos_relation_ratio * num_pos
        num_neg_desired = int(num_neg_desired)
        num_neg_desired = min(num_neg_desired, num_neg)

        negative_sample = sample(list(negatives), num_neg_desired)

        return list(set(positives) | set(negative_sample))

    def ner_training_step(self, example, token_embeddings):
        # print('NER training step')
        candidate_spans = self.ner_subsampler(example.doc._.mentions, example.negative_spans())

        if candidate_spans:

            logits = self.ner(candidate_spans, token_embeddings)
            gold_labels = self.ner.get_gold_labels(candidate_spans, self.entity_type_converter)

            loss = cross_entropy(logits, gold_labels) #nll_loss(logits, gold_labels)
        
            return loss
        else:
            return 0

    def filter_candidate_spans(self, spans):
        spans = [el for el in spans if len(el) < self.max_span_length]
        spans = [el for el in spans if SpanUtils.has_noun(el)]
        return spans

    def ner_inference_step(self, example, token_embeddings):
        # print('NER inference step')
        candidate_spans = example.candidate_spans()
        
        candidate_spans = self.filter_candidate_spans(candidate_spans)

        logits = self.ner(candidate_spans, token_embeddings)
        
        predicted_spans = self.ner.predict(candidate_spans, logits, self.entity_type_converter)
        predicted_mentions = self.ner.filter_nonentities(predicted_spans)
        return predicted_mentions

    def cluster_training_step(self, example, token_embeddings):
        # print('cluster training step')
        
        # candidate_span_pairs = example.positive_span_pairs + example.negative_span_pairs

        candidate_span_pairs = self.coref_subsampler(example.positive_span_pairs(), example.negative_span_pairs(), self.neg_to_pos_span_pair_ratio)
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

    def cluster_inference_step(self, candidate_span_pairs, mentions, token_embeddings, doc):
        # print('cluster inference step')
        if candidate_span_pairs:
            
            logits = self.clusterer(candidate_span_pairs, token_embeddings)

            predicted_span_pairs = self.clusterer.predict(candidate_span_pairs, logits)
            
            predicted_coreferences = self.clusterer.keep_coreferent_pairs(predicted_span_pairs)
            
            predicted_entities = self.clusterer.cluster(mentions, predicted_coreferences, doc)

            return predicted_coreferences, predicted_entities
        else:
            return set(), set()
    
    def rc_training_step(self, example, token_embeddings):

        positive_relations = example.filter_rare_relation_types()

        if positive_relations:
            
            negative_relations = example.negative_relations()
            relations = self.rc_subsampler(positive_relations, negative_relations)

            # relations = positive_relations + negative_relations

            logits = self.rc(relations, token_embeddings)
            gold_labels = self.rc.get_gold_labels(relations, self.relation_type_converter)

            loss = cross_entropy(logits, gold_labels)

            return loss
        else:
            return 0

    def rc_inference_step(self, candidate_relations, token_embeddings):
        # print('RC inference step')

        candidate_relations = list(candidate_relations)
        if candidate_relations:
            logits = self.rc(candidate_relations, token_embeddings)

            predicted_relations = self.rc.predict(candidate_relations, logits, self.relation_type_converter)

            predicted_relations = self.rc.filter_nonrelations(predicted_relations)
            return predicted_relations
        else:
            return set()

    def training_step(self, example, example_index):

        if example_index == 128:
            pass
        # print('training step')

        token_embeddings = self.lm_step(example)

        loss = torch.tensor([0], dtype = torch.float32, requires_grad = True).to(self.device)
        if self.task in ['ner', 'e2e']:
            ner_loss = self.ner_training_step(example, token_embeddings)
            ner_loss *= self.loss_coefficients[0]
            loss += ner_loss
            
        if self.task in ['cluster', 'e2e']:
            clusterer_loss = self.cluster_training_step(example, token_embeddings)
            clusterer_loss *= self.loss_coefficients[1]
            loss += clusterer_loss

        if self.task in ['rc', 'e2e']:
            rc_loss = self.rc_training_step(example, token_embeddings)
            rc_loss *= self.loss_coefficients[2]
            loss += rc_loss

        return loss
    
    def filter_invalid_combination_BioRED(self, relations):
    
        valid_type_combinations = set(('DiseaseOrPhenotypicFeature', 'ChemicalEntity'),
                                ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
                                ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
                                ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
                                ('GeneOrGeneProduct', 'ChemicalEntity'),
                                ('ChemicalEntity', 'ChemicalEntity'),
                                ('ChemicalEntity', 'SequenceVariant')
                                )
        
        valid_type_combinations |= set((el[1], el[0]) for el in valid_type_combinations)

        return [el for el in relations if (el.head.attrs['type'], el.tail.attrs['type']) in valid_type_combinations]


    def inference_step(self, example, example_index):
        # print('inference step')
        # obtain predictions
        token_embeddings = self.lm_step(example)

        # ner
        if self.task in ['ner', 'e2e']:
            predicted_mentions = self.ner_inference_step(example, token_embeddings)
        
        # clustering
        if self.task == 'e2e':
            candidate_span_pairs = self.clusterer.exhaustive_intratype_pairs(predicted_mentions)
            predicted_coreferences, predicted_entities = self.cluster_inference_step(candidate_span_pairs, predicted_mentions, token_embeddings, example.doc)

        elif self.task == 'cluster':
            candidate_span_pairs = example.positive_span_pairs() + example.negative_span_pairs()
            predicted_coreferences, predicted_entities = self.cluster_inference_step(candidate_span_pairs, example.doc._.mentions, token_embeddings, example.doc)

        # rc     
        if self.task == 'e2e':
            candidate_relations = self.rc.dataset2relation_constructor[self.dataset_name](predicted_entities)
            
        elif self.task == 'rc':
            candidate_relations = example.filter_rare_relation_types() + example.negative_relations()
            
        if self.task in ['rc', 'e2e']:
            # candidate_relations = self.filter_invalid_combination_BioRED(candidate_relations)
            predicted_relations = self.rc_inference_step(candidate_relations, token_embeddings)

                
        # example performance
        # print('step performance')
        if self.task in ['ner', 'e2e']:
            ner_scorers = dict()
            for el in self.ner_scorer_classes:
                scorer = el(predicted_mentions, set(example.eval_mentions))
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                ner_scorers[class_name] = scorer
                calculator = self.ner_performance_calculators[class_name]
                calculator.update(counts)
        if self.task in ['cluster', 'e2e']:
            coref_scorers = dict()
            for el in self.coref_scorer_classes:
                scorer = el(predicted_coreferences, example.positive_span_pairs())
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                coref_scorers[class_name] = scorer
                self.coref_performance_calculators[class_name].update(counts)
            entity_scorers = dict()
            for el in self.entity_scorer_classes:
                scorer = el(predicted_entities, example.eval_entities)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                entity_scorers[class_name] = scorer
                self.entity_performance_calculators[class_name].update(counts)
        if self.task in ['rc', 'e2e']:
            rc_scorers = dict()
            for el in self.rc_scorer_classes:
                scorer = el(predicted_relations, example.eval_relations)
                counts = scorer.performance_counts()
                
                class_name = el.__name__
                rc_scorers[class_name] = scorer
                self.rc_performance_calculators[class_name].update(counts)

        # organizing results and details for error analysis
        details = {'text': example.text, 
                   'pmid': example.pmid
                   }
        
        if self.task in ['ner', 'e2e']:
            # details['predicted_mentions'] = predicted_mentions
            details['ner_scorers'] = ner_scorers
        if self.task in ['cluster', 'e2e']:
            # details['predicted_coref_pairs'] = predicted_coreferences
            # details['predicted_entities'] = predicted_entities
            details['coref_scorers'] = coref_scorers
            details['entity_scorers'] = entity_scorers
        if self.task in ['rc', 'e2e']:
            # details['predicted_relations'] = predicted_relations
            details['rc_scorers'] = rc_scorers
        
        return details

    def _update_details(self, details):
        if self.validation_details == []: #1st example of 1st epoch
            self.validation_details.append(details)
        elif isinstance(self.validation_details[0], dict): # 1st epoch
            self.validation_details.append(details)
        else:
            self.validation_details[-1].append(details) # non-1st epoch

    def validation_step(self, example, example_index):
        # if self.current_epoch < -1: #50:
        details = self.inference_step(example, example_index)
        
        self._update_details(details)
        # self.validation_details[-1].append(details)

    def test_step(self, example, example_index):
        details = self.inference_step(example, example_index)
        self.test_details[-1].append(details)

    def _reset_calculators(self, calculator_dict):
        for value in calculator_dict.values():
            value.reset()

    def _compute_calculators(self, calculator_dict):
        return {key: value.compute() for key, value in calculator_dict.items()}

    def _prepare_save_objects(self):
        if not hasattr(self, 'validation_details'):
            self.validation_details = list()
        elif hasattr(self, 'validation_details'):
            if isinstance(self.validation_details[0], dict):
                self.validation_details = [self.validation_details]
            self.validation_details.append(list())

        if not hasattr(self, 'validation_performance'):
            pass
        elif hasattr(self, 'validation_performance'):
            if isinstance(self.validation_performance, dict):
                self.validation_performance = [self.validation_performance]

    def on_validation_start(self):
        self._prepare_save_objects()

    def _get_performance(self):
        performance = dict()
        if self.task in ['ner', 'e2e']:
            performance['ner'] = self._compute_calculators(self.ner_performance_calculators)
        if self.task in ['cluster', 'e2e']:
            performance['coref'] =  self._compute_calculators(self.coref_performance_calculators)
            performance['entity'] =  self._compute_calculators(self.entity_performance_calculators)
        if self.task in ['rc', 'e2e']:
            performance['rc'] =  self._compute_calculators(self.rc_performance_calculators)

        return performance
    
    def _update_performance(self, performance):
        if not hasattr(self, 'validation_performance'):
            self.validation_performance = performance
        elif hasattr(self, 'validation_performance'):
            self.validation_performance.append(performance)

    def _extract_epoch(self):
        match = re.search(r'epoch=(\d+)', self.trainer.ckpt_path)
        epoch_number = int(match.group(1))
        return epoch_number

    def _save_inference_output(self):
        epoch_num = self._extract_epoch()
        
        save_json(self.validation_performance, path.join(self.logger.log_dir, f'performance_epoch{epoch_num}.json'))
        torch.save(self.validation_details, path.join(self.logger.log_dir,f'validation_details{epoch_num}.save'))

    def _reset_all_calculators(self):
        # resetting calculators
        if self.task in ['ner', 'e2e']:
            self._reset_calculators(self.ner_performance_calculators)
        if self.task in ['cluster', 'e2e']:
            self._reset_calculators(self.coref_performance_calculators)
            self._reset_calculators(self.entity_performance_calculators)
        if self.task in ['rc', 'e2e']:
            self._reset_calculators(self.rc_performance_calculators)

    def on_validation_end(self):
        # if self.current_epoch > -1:
        # obtain and save performance
        performance = self._get_performance()
        self._update_performance(performance)
        # self.validation_performance.append(performance)

        print(self.validation_performance)
        
        self._save_inference_output()
        self._reset_all_calculators()

    def configure_optimizers(self):
        print('configuring optimizers')
        lm_optimizer = {'params': self.lm.parameters(), 'lr': self.lm_learning_rate}
        
        optimizer_components = [lm_optimizer]

        if self.task in ['ner', 'e2e']:
            ner_optimizer = {'params': self.ner.parameters(), 'lr': self.learning_rate}
            optimizer_components.append(ner_optimizer)
        if self.task in ['cluster', 'e2e']:
            cluster_optimizer = {'params': self.clusterer.parameters(), 'lr': self.learning_rate}
            optimizer_components.append(cluster_optimizer)
        if self.task in ['rc', 'e2e']:
            rc_optimizer = {'params': self.rc.parameters(), 'lr': self.learning_rate}
            optimizer_components.append(rc_optimizer)

        
        return AdamW(optimizer_components)

    def on_train_start(self):
        self.train_loss = []

    def on_before_backward(self, loss):
        self.train_loss[-1] += loss.item()

    def on_train_epoch_start(self):
        self.train_loss.append(0)


    

# %%
