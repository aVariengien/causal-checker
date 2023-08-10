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

c1_q1_dataset, c2_q2_dataset = get_dataset_pair(
    variables=["city", "character_name", "character_occupation", "season", "day_time"],
    dataset_size=100,
    model=model,
)

c2_q1_dataset = c2_q2_dataset.question_from(c1_q1_dataset)
# %%

c1_q1_dataset = NanoQADataset(
    nb_samples=10,
    tokenizer=model.tokenizer,  # type: ignore
)
# %%
d = evaluate_model(model, c1_q1_dataset, batch_size=10)
print_performance_table(d)
# %%
pprint_nanoqa_prompt(c1_q1_dataset, 0, separator="")
# %%
pprint_nanoqa_prompt(c2_q2_dataset, 0, separator="")
pprint_nanoqa_prompt(c2_q1_dataset, 0, separator="")

# %%
cache = {}
fill_cache(cache, model, c1_q1_dataset, mid_layer=MID_LAYER)


# %%
dla = direct_logit_attribution(
    c1_q1_dataset, model, cache, narrative_variable="correct"
)
dla_mean = dla.mean(dim=-1)
dla_std = dla.std(dim=-1)
px.imshow(dla_mean, aspect="auto")

# %%

df = plot_sorted_dla_bar_chart(
    c1_q1_dataset,
    model,
    cache,
    variable="correct",
)
# %%


plot_num_components(df, np.linspace(0.0, 0.8, 50))


# %%
attn_to_cor_tok = attention_to_nar_var_tok(
    c1_q1_dataset, model, cache, narrative_variable="correct"
)

attn_mean = attn_to_cor_tok.mean(axis=-1)

px.imshow(attn_mean, aspect="auto")

# %%


plot_movers(
    c1_q1_dataset,
    model,
    title="Probed variable: correct",
    mid_layer=MID_LAYER,
    model_name=model_name,
)

# %%

# check if movers are the same no matter the question
variable_datasets = {}
variables = ["city", "character_name", "character_occupation", "season", "day_time"]
for variable in variables:
    variable_datasets[variable] = NanoQADataset(
        name=variable,
        nb_samples=100,
        nb_variable_values=5,
        querried_variables=[variable],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )

# %%
movers_df = {}
for d in variable_datasets.values():
    movers_df[d.name] = get_mover_df(d, model, mid_layer=MID_LAYER)


for d in variable_datasets.values():
    fig = plot_movers(d, model, df=movers_df[d.name], model_name=model_name)
    fig.write_image(f"figs/movers_{model_name}_{d.name}.png")

# %%
movers = {}
percentile = 0.7
for d in variable_datasets.values():
    movers[d.name] = get_movers_names(
        d, model, percentile=percentile, df=movers_df[d.name]
    )

#  compute and print the matrix of intersection of movers for the three different datasets


def plot_intersection_matrix(components_dict: Dict[str, List[Tuple[int, int]]]):
    intersect_matrix = np.zeros((len(components_dict), len(components_dict)))
    for i, (k1, v1) in enumerate(components_dict.items()):
        for j, (k2, v2) in enumerate(components_dict.items()):
            intersect_matrix[i, j] = len(
                set([str(x) for x in v1]).intersection(set([str(x) for x in v2]))
            )

    fig = px.imshow(intersect_matrix, zmin=0)

    # add ticks with the names of the datasets

    fig.update_layout(
        xaxis=dict(
            tickvals=list(range(len(intersect_matrix))),
            ticktext=variables,
        ),
        yaxis=dict(
            tickvals=list(range(len(intersect_matrix))),
            ticktext=variables,
        ),
        title=f"Intersection matrix - percentile = {percentile}",
    )
    fig.show()


# %%

caches_per_qvar = {}
for d in variable_datasets.values():
    caches_per_qvar[d.name] = {}
    fill_cache(caches_per_qvar[d.name], model, d, mid_layer=MID_LAYER)


# %%
# TODO maybe a facet plot here with different color for the bars depending on the type

for d in variable_datasets.values():
    for var in ["city", "character_name", "character_occupation", "season", "day_time"]:
        plot_sorted_dla_bar_chart(
            d,
            model,
            caches_per_qvar[d.name],
            variable=var,
            title_suffix=f" - dataset: {d.name} - var: {var}",
        )
    print(" ==== ")


# %%

# try the steerer experiments city -> character_name

city_dataset = variable_datasets["city"]
char_name_dataset = variable_datasets["character_name"]


city_cache = {}
fill_cache(city_cache, model, city_dataset, mid_layer=MID_LAYER)

char_name_cache = {}
fill_cache(char_name_cache, model, char_name_dataset, mid_layer=MID_LAYER)

chimera_dataset = char_name_dataset.question_from(city_dataset)
chimera_cache = {}
fill_cache(chimera_cache, model, chimera_dataset, mid_layer=MID_LAYER)


df_target_char_name = get_mover_df(
    char_name_dataset, model, narrative_variable="character_name", mid_layer=MID_LAYER
)
df_target_city = get_mover_df(
    char_name_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)

df_source_city = get_mover_df(
    city_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)
# %%
plot_movers(
    char_name_dataset,
    model,
    df=df_target_char_name,
    title="Probed variable: char_name",
    model_name=model_name,
)
# %%
plot_movers(
    char_name_dataset,
    model,
    df=df_target_city,
    title="Probed variable: city",
    model_name=model_name,
)

# %%


# %%
model.reset_hooks()
apply_steering(
    target_dataset=char_name_dataset,
    model=model,
    cache_source=city_cache,
    mid_layer=MID_LAYER,
)

# %%

df_target_char_name_post_steering = get_mover_df(
    char_name_dataset, model, narrative_variable="character_name", mid_layer=MID_LAYER
)
df_target_city_post_steering = get_mover_df(
    char_name_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)
# %%

plot_movers(
    char_name_dataset,
    model,
    df=df_target_char_name_post_steering,
    title="Probed var.: char_name (AFTER STEERING)",
    model_name=model_name,
)
# %%
plot_movers(
    char_name_dataset,
    model,
    df=df_target_city_post_steering,
    title="Probed var.: city (AFTER STEERING)",
    model_name=model_name,
)

# %%
d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)

# %%

chimera_dataset = char_name_dataset.question_from(city_dataset)
d = evaluate_model(
    model, char_name_dataset, batch_size=100, label_nano_qa_dataset=chimera_dataset
)
print_performance_table(d)

# %%

d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)
# %%
model.reset_hooks()
d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)

# %%
plot_movers(
    city_dataset,
    model,
    df=df_source_city,
    title="Probed variable: baseline city dataset",
    model_name=model_name,
)
# %%
model.reset_hooks()
d = evaluate_model(model, city_dataset, batch_size=100)
print_performance_table(d)
# %%


plot_dataframe_comparison(
    df_target_city,
    df_target_city_post_steering,
    name1="City (pre-patching)",
    name2="City (post-patching)",
    variable_name="city",
)
# %%
plot_dataframe_comparison(
    df_target_city,
    df_target_city_post_steering,
    name1="City (pre-patching)",
    name2="City (post-patching)",
    variable_name="city",
)
# %%

plot_dataframe_comparison(
    df_target_char_name,
    df_target_char_name_post_steering,
    name1="Char. name (pre-patching)",
    name2="Char. name (post-patching)",
    variable_name="char_name",
)

# %%
plot_dataframe_comparison(
    df_source_city,
    df_target_city_post_steering,
    name1="City (on the city-dataset)",
    name2="City (post-patching)",
    variable_name="city",
)

# %%
plot_movers(
    char_name_dataset, model, df=df_target_city_post_steering, model_name=model_name
)
# %%

# Let's do attention patching

# %% get the movers (making the union of the city and char_name movers)

char_name_movers = get_movers_names(
    char_name_dataset,
    model,
    percentile=0.0,
    df=df_target_char_name,
    filter_by_layer=False,
    mid_layer=MID_LAYER,
)
city_movers = get_movers_names(
    city_dataset,
    model,
    percentile=0.0,
    df=df_source_city,
    filter_by_layer=False,
    mid_layer=MID_LAYER,
)
all_movers = char_name_movers
for m in city_movers:
    if not m in all_movers:
        all_movers.append(m)


print(f"Number of movers: {len(all_movers)}")
# %%


# %%
model.reset_hooks()

apply_steering(char_name_dataset, model, city_cache, mid_layer=MID_LAYER)
# apply_attention_patch(
#     model,
#     all_movers,
#     chimera_cache,
#     char_name_dataset,
#     source_dataset=chimera_dataset,
#     mode="all_seq",
# )

# %%

logits = model(char_name_dataset.prompts_tok.cuda())
# %%
d = evaluate_model(
    model,
    char_name_dataset,
    logits=logits,
    batch_size=100,
)
print_performance_table(d)

d = evaluate_model(
    model,
    char_name_dataset,
    logits=logits,
    batch_size=100,
    label_nano_qa_dataset=chimera_dataset,
)
print_performance_table(d)
# %%

patched_cache = {}
fill_cache(patched_cache, model, char_name_dataset, mid_layer=MID_LAYER)

# %%
for var in ["character_name", "city"]:
    plot_sorted_dla_bar_chart(
        char_name_dataset,
        model,
        patched_cache,
        variable=var,
        title_suffix=f" {var} (post attn pattern patching)",
    )

# %%
model.reset_hooks()
wt_cache = {}
fill_cache(wt_cache, model, char_name_dataset, mid_layer=MID_LAYER)

# %%
for var in ["character_name", "city"]:
    plot_sorted_dla_bar_chart(
        char_name_dataset,
        model,
        wt_cache,
        variable=var,
        title_suffix=f" {var} (pre-patching)",
    )

# %%
