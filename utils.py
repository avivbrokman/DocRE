# %% libraries
import os
from collections import Counter
import json
from copy import deepcopy
from dataclasses import is_dataclass, replace, fields


import random

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

# def exclusion_deepcopy(instance, exclude_attr = None, new_value = None):
#     kwargs = {f.name: deepcopy(getattr(instance, f.name)) for f in fields(instance) if f.name != exclude_attr}
#     if exclude_attr and new_value is not None:
#         kwargs[exclude_attr] = new_value
#     return type(instance)(**kwargs)

def generalized_replace(instance, **changes):
    if is_dataclass(instance):
        return replace(instance, **changes)
    elif isinstance(instance, list):
        return [replace(el, **changes) for el in instance if is_dataclass(el)]
    elif isinstance(instance, set):
        return {replace(el, **changes) for el in instance if is_dataclass(el)}

# decorator
# def parentless_print(cls):
#     # Save the original __str__ method, if it exists
#     original_str_method = cls.__str__ if '__str__' in cls.__dict__ else None

#     def parentless_str(self):
#         field_strs = []
#         for field in vars(self):
#             if not field.startswith('parent') and not field[0].isupper() and not field == 'annotation':
#                 value = getattr(self, field)
#                 field_strs.append(f"{field}={value}")
#         return f"{cls.__name__}({', '.join(field_strs)})"

#     # Set the new __str__ method for the class
#     cls.__str__ = parentless_str

#     # Return the modified class
#     return cls

# def parentless_print(cls):
#     original_str_method = cls.__str__ if '__str__' in cls.__dict__ else None

#     def parentless_str(self):
#         field_strs = []
#         for field, value in vars(self).items():
#             if not field.startswith('parent') and not field[0].isupper() and field != 'annotation':
#                 field_strs.append(f"{field}={value}")
#         return f"{cls.__name__}({', '.join(field_strs)})"

#     cls.__str__ = parentless_str
#     return cls

# def parentless_print(cls):
#     def parentless_str(self):
#         field_strs = []
#         for field in fields(self):
#             field_name = field.name
#             if not field_name.startswith('parent') and not field_name[0].isupper() and field_name != 'annotation':
#                 value = getattr(self, field_name)
#                 field_strs.append(f"{field_name}={value}")
#         return f"{cls.__name__}({', '.join(field_strs)})"

#     cls.__str__ = parentless_str
#     return cls


def parentless_print(cls):
    def parentless_str(self):
        field_strs = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.startswith('parent') or field.name[0].isupper() or field.name == 'annotation':
                field_strs.append(f"{field.name} = {'present' if value else value}")
            else:
                field_strs.append(f"{field.name} = {value}")
        return f"{cls.__name__}({', '.join(field_strs)})"

    cls.__str__ = parentless_str
    return cls