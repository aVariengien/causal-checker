from .experiments import *
from .plot_fn import *

# %%
import math
import random as rd
import matplotlib.pyplot as plt
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
DATASET_SIZE = 200

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

var = "city"
city_dataset = variable_datasets[var]


all_attn = attention_to_nar_var_tok(
    city_dataset, model, caches_per_qvar[var], narrative_variable=var
)

ef_df, _ = compute_direct_effect_df(
    city_dataset,
    model,
    caches_per_qvar[var],
    narrative_variable=var,
    ref_narrative_variable=[v for v in variables if v != var],
    corrupted_cache=rd_caches_per_qvar[var],
    nb_ressamples=3,
    expand_sample_dim=False,
)

# %% 19,2 - Porto / 17, 31 Cusco/ 17,4 Cusco / 22,9 Cusco / 20,3 Valencia
l = 19
h = 2
name = f"head {l} {h}"
direct_effect = ef_df[ef_df["name"] == name]["all_normalized_direct_effect"].item()
attn = all_attn[l][h]


idx = []
city = "Porto"  # Cusco Busan Valencia Porto Antwerp
for i in range(len(city_dataset)):
    if city_dataset.nanostories[i]["seed"]["city"] == city:
        idx.append(i)


def plot_histogram(data, idx, title, city):
    idx_values = [data[i] for i in idx]
    other_values = [data[i] for i in range(len(data)) if i not in idx]

    plt.figure(figsize=(10, 6))

    # Histogram for other values
    plt.hist(other_values, bins=30, color="blue", alpha=0.6, label="Other cities")

    # Histogram for values at idx
    plt.hist(idx_values, bins=30, color="red", alpha=0.6, label=city)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig("figs/" + title + ".pdf")
    plt.show()


plot_histogram(direct_effect, idx, f"Normalized direct Effect - {name}", city)
plot_histogram(attn, idx, f"Attention probability to the city token - {name}", city)
# %%
