#%% libraries
from os import path
from dataclasses import dataclass, fields
from datasets import load_dataset, concatenate_datasets
import torch
from collections import Counter

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
        print('returning Token')
        if not self.Token:
            print('get parent sentence')
            parent_Sentence = self.parent_sentence.Sentence
            print('instantiating Token')
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
        print('returning sentence')
        if not self.Sentence:
            print('make stub')
            self.Sentence = Sentence.stub()
            print('get parent')
            self.Sentence.parent_example = self.parent_example.Example            
            print('make Tokens')
            self.Sentence.tokens = [el.return_Token() for el in self.words]
            print('get index')
            self.Sentence.index = self.index

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
        self.in_sentence_word_span = (self.annotation['pos'][0], self.annotation['pos'][1] - 1)

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

            head = self.head.return_Cluster()
            tail = self.tail.return_Cluster()
            parent_Example = self.parent_example.Example

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

    def __post_init__(self):
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
            self.Example.clusters = [el.return_Cluster() for el in self.entities]
            self.Example.positive_cluster_pairs = [el.return_ClusterPair() for el in self.relations]            
            

            # self.Example = stub.populate(sentences, clusters, positive_cluster_pairs, parent_Dataset)
            self.Example.process()
            return self.Example
        
#%% DocREDDataset
@parentless_print
@dataclass
class DocREDDataset:
    
    huggingface_dataset: ...
    # relation_type_converter: Optional[dict] = field(default = None)
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
        
        return types
        
    def get_relation_types(self):
        counts = Counter()
        for el_ex in self.examples:
            for el_rel in el_ex.relations:
                counts.update([el_rel.type])
        counts = counts.most_common()
        types = list(zip(*counts))[0]
        
        return types 

    # def _get_relation_type_converter(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     types = [el.type for el in relations]
    #     counts = Counter(types)
    #     save_json(counts, self.counts_save_filename)
        
    #     counts = counts.most_common()
    #     relation_types_by_frequency = list(zip(*counts))[0]

    #     self.relation_type_converter = dict(zip(relation_types_by_frequency, range(len(relation_types_by_frequency))))
    
    # def _convert_relation_types(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     for el in relations:
    #         el.type_index = self.relation_type_converter[el.type]

    # def analyze_relations(self):
    #     relations = unlist([el.relations for el in self.examples])
    #     return [(el.head.type, el.tail.type, el.type, el.type_index) for el in relations]

    def __post_init__(self):
        self._get_examples()
        # if not self.relation_type_converter:
        #     self._get_relation_type_converter()
        # self._convert_relation_types()

    # def return_Dataset(self, tokenizer, entity_types, relation_types):
    #     if not self.Dataset:
    #         examples = [el.return_Example() for el in self.examples]        
    #         self.Dataset = Dataset(examples, tokenizer, entity_types, relation_types)
    #         return self.Dataset
        
    def return_Dataset(self, tokenizer, entity_types, relation_types):
        if not self.Dataset:
            self.Dataset = Dataset.stub(tokenizer, entity_types, relation_types)
            self.Dataset.examples = [el.return_Example() for el in self.examples]
            # stubs = [Example.stub(tokenizer, entity_types, relation_types)] * len(self.examples)
            # examples = [el_ex.populate(el_stub) for el_stub, el_ex in zip(stubs, self.examples)]
            # self.Dataset = stub.populate(examples)
            self.Dataset.process()
            return self.Dataset
    # def return_Dataset(self, tokenizer, entity_types, relation_types):
    #     examples = [el.return_Example() for el in self.examples]        
        
    #     return Dataset(examples, tokenizer, entity_types, relation_types)

#%% processing
# tokenizer
checkpoint = 'bert-base-uncased'
            
# make data paths and directories
output_data_path = path.join('data', 'processed', checkpoint, 'DocRED')

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
print('train')
train_data = DocREDDataset(train_data)
print('valid')
valid_data = DocREDDataset(valid_data)
print('test')
test_data = DocREDDataset(test_data)
print('train + valid')
train_valid_data = DocREDDataset(train_valid_data)

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
train_data = train_data.return_Dataset(tokenizer, entity_types, relation_types)
valid_data = valid_data.return_Dataset(tokenizer, entity_types, relation_types)
test_data = test_data.return_Dataset(tokenizer, entity_types, relation_types)
train_valid_data = train_valid_data.return_Dataset(tokenizer, entity_types, relation_types)


entity_type_converter = train_data.entity_type_converter
relation_type_converter = train_data.relation_type_converter

# Saves everything
torch.save(entity_types, path.join(output_data_path,'entity_types.save'))
torch.save(relation_types, path.join(output_data_path,'relation_types.save'))
torch.save(entity_type_converter, path.join(output_data_path,'entity_type_converter.save'))
torch.save(relation_type_converter, path.join(output_data_path,'relation_type_converter.save'))

torch.save(train_data, path.join(output_data_path,'train_data.save'))    
torch.save(valid_data, path.join(output_data_path,'valid_data.save'))    
torch.save(test_data, path.join(output_data_path,'test_data.save'))   
torch.save(train_valid_data, path.join(output_data_path,'train_valid_data.save'))   

