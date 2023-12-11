#%% libraries
import torch
from collections import Counter

#%%
relations_train = torch.load('/Users/avivbrokman/documents/Kentucky/Grad School/NLP/projects/DocRE/data/processed/DocRED/relation_combinations_train.save')
relations_valid = torch.load('/Users/avivbrokman/documents/Kentucky/Grad School/NLP/projects/DocRE/data/processed/DocRED/relation_combinations_valid.save')
relations_test = torch.load('/Users/avivbrokman/documents/Kentucky/Grad School/NLP/projects/DocRE/data/processed/DocRED/relation_combinations_test.save')

#%%
counter = Counter(relations)

#%% analyze by predicate
# get all predicates
relation_types = list(set(el[2] for el in relations))

i = 1
focal_type = relation_types[i]
dict(el for el in counter.items() if el[0][2] == focal_type)


#%% analyze by type-pair
pairs = list(set((el[0], el[1]) for el in relations))

i = 0
focal_pair = pairs[i]
dict(el for el in counter.items() if (el[0][0], el[0][1]) == focal_pair)

