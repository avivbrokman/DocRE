# %% libraries
import os
from collections import Counter
import importlib
import re
import json
import torch

import random
\
# %% body

def save_json(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def unlist(nested_list):
    unlisted = [subel for el in nested_list for subel in el]
    return unlisted



def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")
    else:
        print(f"Directory already exists at: {dir_path}")

def mode(values):
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [s for s, count in counts.items() if count == max_count]
    return random.choice(modes)  

def apply(fun):
    
    def list_version_of_fun(_list):
        return [fun(el) for el in _list]

    return list_version_of_fun