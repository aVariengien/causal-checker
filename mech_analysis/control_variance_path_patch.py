from .experiments import *
from .plot_fn import *

# %%
import math
import random as rd
from functools import partial
from pprint import pprint
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from swap_graphs.core import WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    pprint_nanoqa_prompt,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import (
    check_tokenizer_nanoQA,
    print_performance_table,
)
from swap_graphs.utils import clean_gpu_mem, printw
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # Hooking utilities

from experiments import *  # type: ignore
from plot_fn import *  # type: ignore
from transformer_lens.utils import test_prompt

torch.set_grad_enabled(False)

# %%

model_name = "pythia-2.8b"
model = HookedTransformer.from_pretrained(
    model_name=model_name, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
)

MID_LAYER = None
if model_name == "pythia-2.8b":
    MID_LAYER = 16
else:
    raise ValueError(f"no mid layer set for {model_name}")

# %%
DATASET_SIZE = 10

variable_datasets = {}
caches_per_qvar = {}
variables = ["city", "character_name", "character_occupation"]
complement_datasets = {}
for variable in variables:
    variable_datasets[variable] = NanoQADataset(
        name=variable,
        nb_samples=DATASET_SIZE,
        nb_variable_values=5,
        querried_variables=[variable],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )

    complement_datasets[variable] = NanoQADataset(
        name=variable,
        nb_samples=DATASET_SIZE,
        nb_variable_values=5,
        querried_variables=[v for v in variables if v != variable],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )
# %%
rd_caches_per_qvar = {}
for variable in variables:
    caches_per_qvar[variable] = {}
    fill_cache(
        caches_per_qvar[variable],
        model,
        variable_datasets[variable],
        mid_layer=MID_LAYER,
    )
    rd_caches_per_qvar[variable] = {}
    fill_cache(
        rd_caches_per_qvar[variable],
        model,
        complement_datasets[variable],
        mid_layer=MID_LAYER,
        randomize=True,
    )


# %%
def compute_overlap_mtx_topK(pp1, pp2, K=50):
    intersection_sizes = []

    for i in range(pp1.shape[-1]):
        # Flatten and get top K indices for pp1 and pp2 at slice i
        _, indices_pp1 = torch.topk(pp1[:, :, i].flatten(), K)
        _, indices_pp2 = torch.topk(pp2[:, :, i].flatten(), K)

        # Calculate the intersection of the indices and its size
        intersection = torch.tensor(
            list(set(indices_pp1.tolist()) & set(indices_pp2.tolist()))
        )
        intersection_size = len(intersection) / K

        intersection_sizes.append(intersection_size)

    return intersection_sizes


def compute_overlap_mtx_threshold(pp1, pp2, threshold=0.5):
    intersection_sizes = []

    for i in range(pp1.shape[-1]):
        # Get indices where absolute values are above threshold for pp1 and pp2 at slice i
        flat_pp1 = pp1[:, :, i].flatten()
        flat_pp2 = pp2[:, :, i].flatten()

        indices_pp1 = torch.where(torch.abs(flat_pp1) > threshold)[0]
        indices_pp2 = torch.where(torch.abs(flat_pp2) > threshold)[0]

        # Calculate the intersection of the indices and its size
        intersection = torch.tensor(
            list(set(indices_pp1.tolist()).intersection(set(indices_pp2.tolist())))
        )
        intersection_size = len(set(intersection)) / len(
            set(indices_pp1.tolist()).union(set(indices_pp2.tolist()))
        )
        print(
            len(intersection),
            len(indices_pp1),
            len(indices_pp2),
            len(list(set(indices_pp1).union(set(indices_pp2)))),
        )

        intersection_sizes.append(intersection_size)
        # print(indices_pp1, indices_pp2)

    return intersection_sizes


nb_ressample = 20

intersections = []
for var in variables:
    print(var)
    pp1 = path_patching_logits(
        variable_datasets[var],
        model,
        caches_per_qvar[var],
        corrupted_cache=rd_caches_per_qvar[var],
        narrative_variable=var,
        ref_narrative_variable=None,
        probs=False,
        nb_ressamples=nb_ressample,
    )
    print("mid")
    pp2 = path_patching_logits(
        variable_datasets[var],
        model,
        caches_per_qvar[var],
        corrupted_cache=rd_caches_per_qvar[var],
        narrative_variable=var,
        ref_narrative_variable=None,
        probs=False,
        nb_ressamples=nb_ressample,
    )
    new_inter = compute_overlap_mtx_threshold(pp1, pp2, threshold=0.03)

    print(np.mean(new_inter))
    intersections += new_inter


print(np.mean(intersections))

# %%
