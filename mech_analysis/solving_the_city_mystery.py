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

torch.set_grad_enabled(False)


# %%
model_name = "pythia-2.8b"
model = HookedTransformer.from_pretrained(
    model_name=model_name, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
)

MID_LAYER = None
if model_name == "pythia-2.8b":
    MID_LAYER = 17
else:
    raise ValueError(f"no mid layer set for {model_name}")


# %%

nano_qa_dataset = NanoQADataset(
    nb_samples=10,
    tokenizer=model.tokenizer,  # type: ignore
    querried_variables=["city"],
    nb_variable_values=5,
)

# %%
from transformer_lens.utils import test_prompt


for i in range(len(nano_qa_dataset)):
    test_prompt(nano_qa_dataset.prompts_text[i], " bl", model, prepend_bos=False)
# %%

d = evaluate_model(model, nano_qa_dataset, batch_size=10)
print_performance_table(d)
# %%

var = "city"
alt_var = None
error_bars = True

mono_sample_dataset = NanoQADataset(
    nb_samples=100,
    tokenizer=model.tokenizer,  # type: ignore
    querried_variables=[var],
    nb_variable_values=5,
)
mono_sample_dataset.name = f"MonoQ - {var}"
mono_cache = {}
fill_cache(mono_cache, model, mono_sample_dataset, mid_layer=MID_LAYER)
df = plot_sorted_dla_bar_chart(
    mono_sample_dataset,
    model,
    mono_cache,
    variable=var,
    alt_variable=alt_var,
    error_bar=error_bars,
    title_suffix=f" - dataset: {mono_sample_dataset.name}",
)


# filter for city
# %%
city_specific_stories = []
questions = []

filtered_mono_sample_dataset = NanoQADataset(
    nb_samples=100,
    tokenizer=model.tokenizer,  # type: ignore
    querried_variables=[var],
    nb_variable_values=5,
)

city_name = "Porto"


filtered_mono_sample_dataset.name = f"Mono city - {city_name}"
for i in range(len(filtered_mono_sample_dataset)):
    if filtered_mono_sample_dataset.nanostories[i]["seed"]["city"] == city_name:
        city_specific_stories.append(filtered_mono_sample_dataset.nanostories[i])
        questions.append(filtered_mono_sample_dataset.questions[i])
filtered_mono_sample_dataset.nb_samples = len(city_specific_stories)
filtered_mono_sample_dataset.build_from_questions_and_stories(
    questions=questions, nanostories=city_specific_stories
)

print(f"nb samples: {len(filtered_mono_sample_dataset)}")
filtered_mono_cache = {}
fill_cache(
    filtered_mono_cache, model, filtered_mono_sample_dataset, mid_layer=MID_LAYER
)

df = plot_sorted_dla_bar_chart(
    filtered_mono_sample_dataset,
    model,
    filtered_mono_cache,
    variable=var,
    alt_variable=alt_var,
    error_bar=True,
    title_suffix=f" - dataset: {filtered_mono_sample_dataset.name}",
)

# pprint_nanoqa_prompt(mono_sample_dataset, 0, separator="")


# %%

dla = direct_logit_attribution(
    mono_sample_dataset,
    model,
    mono_cache,
    narrative_variable="character_occupation",
)

# %%

plt.hist(dla.sum(dim=0).sum(dim=0), bins=50)

# %%
mean_abs_dla = dla.abs().mean(dim=-1)
flattened_arr = mean_abs_dla.ravel()
sorted_indices = np.argsort(flattened_arr)

# Take the top 10 indices
top_10_indices = sorted_indices[-20:]
# Convert the flattened indices to the original indices in the 2D array
row_indices, col_indices = np.unravel_index(top_10_indices, mean_abs_dla.shape)
indices = list(zip(row_indices, col_indices))[::-1]

# %%

for k in range(len(indices) // 3):
    for i, j in indices[k * 3 : k * 3 + 3]:
        plt.hist(dla[i, j], bins=50, alpha=0.5, label=f"{i}_{j}")
    plt.legend()
    plt.show()

# %%
for i, j in indices[3:6]:
    plt.hist(dla[i, j], bins=50, alpha=0.5)
# %%
indices_filtered = [(20, 3), (19, 2), (22, 25)]

indices_dict = {}

reverse = {"19_2", "22_7", "20_25", "17_18"}
check_abs_value = True

for i, j in indices:
    if (i, j) not in indices_filtered:
        continue
    threshold = (dla[i, j].quantile(0.8) + dla[i, j].quantile(0.2)) / 2
    name = str(i) + "_" + str(j)

    if not check_abs_value:
        if name in reverse:
            indices_dict[str(i) + "_" + str(j)] = list(
                np.where(dla[i, j] < threshold)[0]
            )
        else:
            indices_dict[str(i) + "_" + str(j)] = list(
                np.where(dla[i, j] > threshold)[0]
            )
    else:
        idces = list(np.where(np.abs(dla[i, j]) > 0.2)[0])
        if len(idces) < 60:
            indices_dict[str(i) + "_" + str(j)] = idces


# for city in ["Valencia", "Porto", "Antwerp", "Cusco", "Busan"]:
#     indices_dict[city] = [
#         i
#         for i in range(len(mono_sample_dataset))
#         if mono_sample_dataset.nanostories[i]["seed"]["city"] == city
#     ]

for job in ["architect", "astronomer", "veterinarian", "librarian", "florist"]:
    indices_dict[job] = [
        i
        for i in range(len(mono_sample_dataset))
        if mono_sample_dataset.nanostories[i]["seed"]["character_occupation"] == job
    ]

plot_intersection_mtx(indices_dict, title="Indices of abs DLA>0.1")
# plot in

# %%

plt.hist(dla[19, 2], bins=50)
# %%
row = dla[19, 2]
n = 0
for i in range(len(row)):
    if row[i] < 1.0:
        if n < 20:
            n += 1
            pprint_nanoqa_prompt(mono_sample_dataset, i, separator="")

# %%


# %%
attn_to_cor_tok = attention_to_nar_var_tok(
    mono_sample_dataset, model, mono_cache, narrative_variable="correct"
)
# %%

l, h = 19, 2

plt.hist(
    attn_to_cor_tok[l, h],
    bins=50,
)
plt.title(f"Head {l} {h} attention distribution")

# %%
