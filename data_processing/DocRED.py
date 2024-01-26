#%% libraries
from os import path
from dataclasses import dataclass, fields
from datasets import load_dataset, concatenate_datasets
import torch
from collections import Counter
from copy import deepcopy

from transformers import AutoTokenizer

from utils import make_dir, unlist, mode, parentless_print

from data_classes import Token, Span, Sentence, Cluster, ClusterPair, Example, Dataset

#%% DocREDWord
@parentless_print
@dataclass
class DocREDWord:
    annotation: str
    in_sentence_index: int
    parent_sentence: ...
    Token: Token = None

    def _get_index(self):
        previous_sentences = self.parent_sentence.parent_example.sentences
        self.index = sum([len(el) for el in previous_sentences]) + self.in_sentence_index

    def __post_init__(self):
        self._get_index()

    def return_Token(self):
        # print('returning Token')
        if not self.Token:
            # print('get parent sentence')
            parent_Sentence = self.parent_sentence.Sentence
            # print('instantiating Token')
            self.Token = Token(self.annotation, self.in_sentence_index, self.index, parent_Sentence)
            return self.Token
        
#%% DocREDSentence
@parentless_print
@dataclass
class DocREDSentence:
    annotation: list[str]
    index: int
    parent_example: ...
    Sentence: Sentence = None

    def _get_words(self):
        self.words = [DocREDWord(el, i, self) for i, el in enumerate(self.annotation)]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

    def __post_init__(self):
        self._get_words()

    def return_Sentence(self):
        # print('returning sentence')
        if not self.Sentence:
            # print('make stub')
            self.Sentence = Sentence.stub()
            # print('get parent')
            self.Sentence.parent_example = self.parent_example.Example

            # print('get index')
            self.Sentence.index = self.index
            
            # print('make Tokens')
            self.Sentence.tokens = [el.return_Token() for el in self.words]

            # self.Sentence =  Sentence(tokens, self.index, parent_Example)
            return self.Sentence

#%% DocREDMention
@parentless_print
@dataclass
class DocREDMention:

    annotation: ...
    
    parent_entity: ...
    Span: Span = None

    def _get_string(self):
        self.string = self.annotation['name']

    def _get_in_sentence_word_span(self):
        self.in_sentence_word_span = tuple(self.annotation['pos'])

    def _get_sentence_index(self):
        self.sentence_index = self.annotation['sent_id']

    def _get_type(self):
        self.type = self.annotation['type']

    def _get_words(self):
        sentence = self.parent_entity.parent_example.sentences[self.sentence_index]
        self.words = [sentence[i] for i in range(*self.annotation['pos'])]

    def _get_span(self):
        self.span = (self.words[0].index, self.words[-1].index)

    def __post_init__(self):
        self._get_string()
        self._get_sentence_index()
        self._get_in_sentence_word_span()
        self._get_type()
        self._get_words()
        self._get_span()

    def return_Span(self):
        if not self.Span:
            self.Span = Span.stub()
            parent_example = self.parent_entity.parent_example.Example
            sentence = parent_example.sentences[self.sentence_index]
            self.Span.tokens = sentence.tokens[self.in_sentence_word_span[0]:self.in_sentence_word_span[1]]
            
            self.Span.type = self.type
            # self.Span = Span(tokens, self.type)
            self.Span.process()
            self.Span._get_subword_indices()
            return self.Span

#%% DocREDEntity
@parentless_print
@dataclass
class DocREDEntity:

    annotation: ...

    parent_example: ...
    Cluster: Cluster = None

    def _get_mentions(self):
        self.mentions = list(DocREDMention(el, self) for el in self.annotation)

    def _get_type(self):
        self.type = mode([el.type for el in self.mentions])

    def __hash__(self):
        
        info = [el.span for el in self.mentions]
        info.append(self.type)

        return hash(tuple(info))

    def __post_init__(self):
        self._get_mentions()
        self._get_type()

    # # def return_Cluster(self, class_converter):
    # def return_Cluster(self):
    #     if not self.Cluster:
            
            
    #         spans = set(el.return_Span() for el in self.mentions)
    #         parent_Example = self.parent_example.Example

    #         self.Cluster = Cluster(spans, parent_Example, self.type)

    #         return self.Cluster
        
    def return_Cluster(self):
        if not self.Cluster:
            self.Cluster = Cluster.stub()
            
            self.Cluster.spans = set(el.return_Span() for el in self.mentions)
            self.Cluster.parent_example = self.parent_example.Example
            self.Cluster.type = self.type

            # self.Cluster = stub.populate(spans, parent_Example, self.type)
            self.Cluster.process()
        return self.Cluster

#%% DocREDRelation
@parentless_print
@dataclass
class DocREDRelation:
    
    head_index: int
    tail_index: int
    type: str

    parent_example: ...
    ClusterPair: ClusterPair = None
    
    def _get_entities(self):
        # print('entities: ', self.parent_example.entities)
        self.head = self.parent_example.entities[self.head_index]
        self.tail = self.parent_example.entities[self.tail_index]

    def __post_init__(self):
        self._get_entities()

    def return_ClusterPair(self):
        if not self.ClusterPair:
            # self.ClusterPair = ClusterPair.stub()
            # self.ClusterPair.head = self.head.return_Cluster()
            # self.ClusterPair.tail = self.tail.return_Cluster()
            # self.ClusterPair.type = self.type
            # self.ClusterPair.parent_example = self.parent_example.Example

            # print('head: ', self.head.Cluster)

            head = self.head.return_Cluster()
            tail = self.tail.return_Cluster()
            parent_Example = self.parent_example.Example

            # print('head: ', head)

            self.ClusterPair = ClusterPair(head, tail, parent_Example, self.type)
        return self.ClusterPair
#%% DocREDExample
@parentless_print
@dataclass
class DocREDExample:

    annotation: ...
    parent_dataset: ...
    Example: Example = None


    '''
    tokens
    
    candidate_spans
    candidate_span_pairs
    candidate_cluster_pairs
    
    full_clusters     
    cluster_relations    
    
    span_labels
    cluster_labels
    relation_labels
    '''

    def _get_title(self):
        self.title = self.annotation['title']

    def _get_sentences(self):
        raw_sentences = self.annotation['sents']
        self.sentences = list()
        for i, el in enumerate(raw_sentences):
            self.sentences.append(DocREDSentence(el, i, self))

    def _get_words(self):
        self.words = unlist([el.words for el in self.sentences])           
    
    def _get_entities(self):
        self.entities = [DocREDEntity(el, self) for el in self.annotation['vertexSet']]

    def _get_relations(self):
        labels = self.annotation['labels']
        triples = list(zip(labels['head'], labels['tail'], labels['relation_text']))
        
        self.relations = [DocREDRelation(el[0], el[1], el[2], self) for el in triples]

    def remove_invalid_tokens(self, invalid_tokens = ['\xa0', ' \xa0', ' ']):

        def invalid_in_sentence_indices(sentence):
            temp = [i for i, el in enumerate(sentence) if el in invalid_tokens]
            temp.sort(reverse = True) # temp
            # return [i for i, el in enumerate(sentence) if el in invalid_tokens]
            return temp #temp
        
            
        def remove_invalid_tokens(sentences):
            def workhorse(sentence):
                return [el for el in sentence if el not in invalid_tokens]
            
            new = [workhorse(el) for el in sentences] # temp
            return new # temp
            # return [workhorse(el) for el in sentences]
        
        def adjust_span_indices(token_index, sentence_index):
            for el_entity in self.annotation['vertexSet']:
                for el_span in el_entity:
                    if el_span['sent_id'] == sentence_index:
                        if el_span['pos'][0] > token_index:
                            el_span['pos'][0] -= 1
                        if el_span['pos'][1] > token_index:
                            el_span['pos'][1] -= 1
                        if el_span['pos'][0] == el_span['pos'][1]:
                            raise ValueError('Span has length 0')
                        
        def adjust_span_strings(entities):
            def entity_apply(entity):
                for el in entity:
                    mention_workhorse(el)
            
            def mention_workhorse(mention):
                mention['name'] = mention['name'].replace('\xa0', '').replace('  ', ' ').strip()
                
            for el in entities:
                entity_apply(el)
                
            
                
        invalid_indices = [invalid_in_sentence_indices(el) for el in self.annotation['sents']]
        
        self.annotation['sents'] = remove_invalid_tokens(self.annotation['sents'])
        
        for i_sent, el_invalid_indices in enumerate(invalid_indices):
            for el_index in el_invalid_indices:
                adjust_span_indices(el_index, i_sent)
                
        adjust_span_strings(self.annotation['vertexSet'])

    def _clean(self):
        self.remove_invalid_tokens()
        

    def __post_init__(self):
        if self.annotation['title'] == 'El Escorial, Madrid': #temp
            pass
        self._clean()
        self._get_title()
        self._get_sentences()
        self._get_words()
        self._get_entities()
        self._get_relations()

    # def return_Example(self):
    #     if not self.Example:
    #         sentences = [el.return_Sentence() for el in self.sentences]
    #         clusters = [el.return_Cluster() for el in self.entities]
    #         positive_cluster_pairs = [el.return_ClusterPair() for el in self.relations]
    #         parent_Dataset = self.parent_dataset.Dataset
            
    #         self.Example = Example(sentences, clusters, positive_cluster_pairs, parent_Dataset)

    #         return self.Example
        
    def return_Example(self):
        if not self.Example:
            self.Example = Example.stub()
            # print(self.sentences)
            # print('fields: ', fields(self.sentences[0]))
            self.Example.parent_dataset = self.parent_dataset.Dataset
            self.Example.sentences = [el.return_Sentence() for el in self.sentences]
            self.Example.clusters = set(el.return_Cluster() for el in self.entities)
            self.Example.positive_cluster_pairs = set(el.return_ClusterPair() for el in self.relations)  
            


            # print('entities: ', self.entities, '/n')
            # print('cluster1: ', self.entities[0].Cluster, '/n')
            # print('cluster1: ', self.Example.clusters, '/n')
            
            # print('cluster pair: ', self.Example.positive_cluster_pairs.pop())
            # print('cluster pairs: ', self.Example.positive_cluster_pairs.pop())

            # self.Example = stub.populate(sentences, clusters, positive_cluster_pairs, parent_Dataset)
            self.Example.process()
            return self.Example
        
#%% DocREDDataset
@parentless_print
@dataclass
class DocREDDataset:
    
    huggingface_dataset: ...
    # relation_class_converter: Optional[dict] = field(default = None)
    # counts_save_filename: Optional[dict] = field(default = 'data/processed/DocRED/relation_type_counts.json')
    Dataset: Dataset = None
                
    def _get_examples(self):
        self.examples = [DocREDExample(el, self) for el in self.huggingface_dataset]

    def get_entity_types(self):
        counts = Counter()
        for el_ex in self.examples:
            for el_ent in el_ex.entities:
                counts.update([el_ent.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        return list(types)
        
    def get_relation_types(self):
        counts = Counter()
        for el_ex in self.examples:
            for el_rel in el_ex.relations:
                counts.update([el_rel.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        

        return list(types) 
    # def _get_relation_class_converter(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     types = [el.type for el in relations]
    #     counts = Counter(types)
    #     save_json(counts, self.counts_save_filename)
        
    #     counts = counts.most_common()
    #     relation_types_by_frequency = list(zip(*counts))[0]

    #     self.relation_class_converter = dict(zip(relation_types_by_frequency, range(len(relation_types_by_frequency))))
    
    # def _convert_relation_types(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     for el in relations:
    #         el.type_index = self.relation_class_converter[el.type]

    # def analyze_relations(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     return [(el.head.type, el.tail.type, el.type, el.type_index) for el in relations]

    def __post_init__(self):
        self._get_examples()
        # if not self.relation_class_converter:
        #     self._get_relation_class_converter()
        # self._convert_relation_types()

    # def return_Dataset(self, tokenizer, entity_types, relation_types):
    #     if not self.Dataset:
    #         examples = [el.return_Example() for el in self.examples]        
    #         self.Dataset = Dataset(examples, tokenizer, entity_types, relation_types)
    #         return self.Dataset

    def filter_by_title(self, title):
        self.examples = [el for el in self.examples if el.title != title]

    def return_Dataset(self, tokenizer, max_length, entity_types, relation_types):
        if not self.Dataset:
            self.Dataset = Dataset.stub(tokenizer, max_length, entity_types, relation_types)
            self.Dataset.process()

            self.Dataset.examples = [el.return_Example() for el in self.examples]

            self.Dataset._get_subword_indices(self.Dataset)


            
            # stubs = [Example.stub(tokenizer, entity_types, relation_types)] * len(self.examples)
            # examples = [el_ex.populate(el_stub) for el_stub, el_ex in zip(stubs, self.examples)]
            # self.Dataset = stub.populate(examples)
            return self.Dataset
    # def return_Dataset(self, tokenizer, entity_types, relation_types):
    #     examples = [el.return_Example() for el in self.examples]        
        
    #     return Dataset(examples, tokenizer, entity_types, relation_types)

#%% processing
# tokenizer
checkpoint = 'bert-base-uncased'
max_length = 10
            
# make data paths and directories
output_data_path = path.join('data', 'processed', 'DocRED', checkpoint)

make_dir(output_data_path)

# load dataset from hugginface
huggingface_dataset = load_dataset('docred')

# make all splits of dataset
train_data = huggingface_dataset['train_annotated']
valid_data = huggingface_dataset['validation']
test_data = huggingface_dataset['test']

train_valid_data = concatenate_datasets([train_data, valid_data])
full_data = concatenate_datasets([train_data, valid_data, test_data])

# processes full dataset to get complete list of entity types and relation types
full_data = DocREDDataset(full_data)
entity_types = full_data.get_entity_types()
relation_types = full_data.get_relation_types()


# processes all datasets
print('parsing train dataset')
train_data = DocREDDataset(train_data)
print('parsing validation dataset')
valid_data = DocREDDataset(valid_data)
print('parsing test dataset')
test_data = DocREDDataset(test_data)
# print('parsing train + validation dataset')
# train_valid_data = DocREDDataset(train_valid_data)

# filtering poorly annotated examples
# train_data.filter_by_title('The Secret of the Nagas')
# valid_data.filter_by_title('Paul Morphy')


#%% More processing
# relation_combinations_train = train_data.analyze_relations()
# relation_combinations_valid = valid_data.analyze_relations()
# relation_combinations_test = test_data.analyze_relations()
# torch.save(relation_combinations_train, os.path.join(output_data_path, 'relation_combinations_train.save'))
# torch.save(relation_combinations_valid, os.path.join(output_data_path, 'relation_combinations_valid.save'))
# torch.save(relation_combinations_test, os.path.join(output_data_path, 'relation_combinations_test.save'))

#%% More processing
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# converts processed datasets into modeling datasets
print('making train modeling dataset')
train_data = train_data.return_Dataset(tokenizer, max_length, deepcopy(entity_types), deepcopy(relation_types))
print('making validation modeling dataset')
valid_data = valid_data.return_Dataset(tokenizer, max_length, deepcopy(entity_types), deepcopy(relation_types))
print('making test modeling dataset')
test_data = test_data.return_Dataset(tokenizer, max_length, deepcopy(entity_types), deepcopy(relation_types))
# print('making train + validation modeling dataset')
# train_valid_data = train_valid_data.return_Dataset(tokenizer, max_length, entity_types, relation_types)


entity_class_converter = train_data.entity_class_converter
relation_class_converter = train_data.relation_class_converter

# Saves everything
torch.save(entity_types, path.join(output_data_path,'entity_types.save'))
torch.save(relation_types, path.join(output_data_path,'relation_types.save'))
torch.save(entity_class_converter, path.join(output_data_path,'entity_class_converter.save'))
torch.save(relation_class_converter, path.join(output_data_path,'relation_class_converter.save'))

torch.save(train_data, path.join(output_data_path,'train_data.save'))    
torch.save(valid_data, path.join(output_data_path,'validation_data.save'))    
torch.save(test_data, path.join(output_data_path,'test_data.save'))   
# torch.save(train_valid_data, path.join(output_data_path,'train_validation_data.save'))   

