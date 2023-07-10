# %%
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import torch
from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    Attribute,
    Entity,
    OperationDataset,
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
from causal_checker.datasets.dataset_utils import gen_dataset_family
from functools import partial
import random as rd
import json


def gen_counterfact_prompt():
    pass


# %%
with open("../../data/counterfact.json") as f:
    d = json.loads(f.read())
# %%]


# %%
def get_sorted_freq(d, key):
    freq = {}
    for fact in d:
        fact = fact["requested_rewrite"]
        if key in fact:
            if fact[key] not in freq:
                freq[fact[key]] = 1
            else:
                freq[fact[key]] += 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)


# %%

subjects = get_sorted_freq(d, "subject")
most_freq_subjects = [s for s, n in subjects if n >= 3]  # more than 3 relations
relations = {}
for fact in d:
    fact = fact["requested_rewrite"]
    if fact["subject"] in most_freq_subjects:
        if fact["subject"] not in relations:
            relations[fact["subject"]] = [fact["relation_id"]]
        else:
            relations[fact["subject"]].append(fact["relation_id"])


# %% find the three most frequent relations for each subject
def unique_order(l) -> str:
    l = list(l)
    return ",".join(sorted(list(set(l))))


# %%
all_relations = [
    set(l.split(",")) for l in set([unique_order(r) for s, r in relations.items()])
]

triplet_instances = {unique_order(k): set() for k in all_relations}

for fact in d:
    fact = fact["requested_rewrite"]
    subject = fact["subject"]
    if subject in most_freq_subjects:
        for r in all_relations:
            if r.issubset(relations[subject]):
                triplet_instances[unique_order(r)].add(subject)

# %%

for k, v in triplet_instances.items():
    if len(k.split(",")) >= 3:
        print(len(v), k, v)

# %%
relations = {}
subjects = {}
relations_prompts = {}
for fact in d:
    relation_id = fact["requested_rewrite"]["relation_id"]
    if relation_id not in relations:
        relations[relation_id] = 1
        relations_prompts[relation_id] = [fact["requested_rewrite"]["prompt"]]
    else:
        relations[relation_id] += 1
        relations_prompts[relation_id].append(fact["requested_rewrite"]["prompt"])


# %%
sorted_relations = sorted(relations.items(), key=lambda x: x[1], reverse=True)

for r, n in sorted_relations:
    print(r, n, relations_prompts[r])
# %%
