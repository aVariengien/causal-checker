# Goal: motivate the slip by function
# show the weird behavior for the city variable

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

DATASET_SIZE = 50
variable_datasets = {}
baseline_datasets = {}
baseline_cache = {}
caches_per_qvar = {}
variables = ["city", "character_name", "character_occupation", "season", "day_time"]
for variable in variables:
    variable_datasets[variable] = NanoQADataset(
        name=variable,
        nb_samples=DATASET_SIZE,
        nb_variable_values=5,
        querried_variables=[variable],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )
    baseline_datasets[variable] = NanoQADataset(
        name="baseline " + variable,
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
        variable_datasets[variable],
        mid_layer=MID_LAYER,
        randomize=True,
    )
    baseline_cache[variable] = {}
    fill_cache(
        baseline_cache[variable],
        model,
        baseline_datasets[variable],
        mid_layer=MID_LAYER,
    )
# %%

direct_effect_df = {}
for variable in variables:
    print(f"Variable: {variable}")
    direct_effect_df[variable], _ = compute_direct_effect_df(
        dataset=variable_datasets[variable],
        model=model,
        cache=caches_per_qvar[variable],
        narrative_variable=variable,
        ref_narrative_variable=[v for v in variables if v != variable],
        corrupted_cache=baseline_cache[variable],
        nb_ressamples=20,
        expand_sample_dim=True,
    )
    direct_effect_df[variable]["narrative_variable"] = variable

# %%
big_dataframe = pd.concat(
    [df.assign(narrative_variable=key) for key, df in direct_effect_df.items()],
    ignore_index=True,
)

# %%

source_df = big_dataframe

dataset_name = "All questions"

mean_std_direct_effect_df = (
    source_df.groupby("name")["direct_effect"].agg(["mean", "std"]).reset_index()
)
mean_std_direct_effect_df.rename(
    columns={"mean": "direct_effect_mean", "std": "direct_effect_std"}, inplace=True
)
mean_std_direct_effect_df["abs_direct_effect_mean"] = mean_std_direct_effect_df[
    "direct_effect_mean"
].abs()


total_abs_direct_effect = source_df["ref_ld"].mean() - source_df["cor_ld"].mean()


mean_std_direct_effect_df["relative_direct_effect_mean"] = (
    mean_std_direct_effect_df["direct_effect_mean"] / total_abs_direct_effect
)
mean_std_direct_effect_df["relative_direct_effect_std"] = (
    mean_std_direct_effect_df["direct_effect_std"] / total_abs_direct_effect
)

percent = 0.03
df = mean_std_direct_effect_df
error_bar = True
df_sorted = df.sort_values("abs_direct_effect_mean", ascending=False)
threshold = df_sorted["abs_direct_effect_mean"].quantile(1 - percent)
df_filtered = df_sorted[df_sorted["abs_direct_effect_mean"] > threshold]

x = df_filtered["name"]
y = df_filtered["relative_direct_effect_mean"]

fig = go.Figure()

if error_bar:
    error = df_filtered["relative_direct_effect_std"]
    fig.add_trace(
        go.Bar(x=x, y=y, error_y=dict(type="data", array=error), name="Relative")
    )
else:
    fig.add_trace(go.Bar(x=x, y=y, name="Mean"))

fig.update_layout(
    title=f"Top {int(percent*100)}% of largest absolute direct effect Entries (Total {total_abs_direct_effect:.2f}) {dataset_name}",
    xaxis=dict(title="Name"),
    yaxis=dict(title="Normalized Direct Effect"),
    xaxis_tickangle=-90,
    width=1200,
    height=800,
)

fig.show()

fig.write_image(f"figs/top_3_percent_all_vars_{dataset_name}.pdf")

# show in matrix form

effect = np.zeros((model.cfg.n_layers, model.cfg.n_heads + 1))
for h in range(model.cfg.n_heads + 1):
    for l in range(model.cfg.n_layers):
        if h >= model.cfg.n_heads:
            name = f"mlp {l}"
        else:
            name = f"head {l} {h}"
        effect[l, h] = mean_std_direct_effect_df[
            mean_std_direct_effect_df["name"] == name
        ]["relative_direct_effect_mean"].iloc[0]


fig = show_mtx(
    effect,
    model,
    title=f"Normalized Direct Effect ({dataset_name})",
    color_map_label="Normalized Direct Effect",
    return_fig=True,
)
assert fig is not None

fig.update_layout(
    width=800,
    height=800,
)
fig.show()
fig.write_image(f"figs/attribution_matrix_{dataset_name}.pdf")

# %%


# %%


# %%
K = None
threshold = 0.03

print(f"overlap between top {K} components")

m, s = compute_average_overlap(big_dataframe, N=1000, K=K, threshold=threshold)
print(f"total : {m:.2f} +- {s:.2f}")


for variable in ALL_NAR_VAR:
    direct_effect_df[variable]["narrative_variable"] = variable
    m, s = compute_average_overlap(
        direct_effect_df[variable], N=1000, K=K, threshold=threshold
    )
    print(f"Variable: {variable}, {m:.2f} +- {s:.2f}")
# %%


def std_by_group(df, K=5):
    total_effect = df["ref_ld"].mean() - df["cor_ld"].mean()

    normalized_std_by_group = df.groupby(["name"])["direct_effect"].std().reset_index()
    normalized_std_by_group["direct_effect"] = (
        normalized_std_by_group["direct_effect"] / total_effect
    )

    mean_effect = df.groupby(["name"])["direct_effect"].mean().reset_index()
    topK = mean_effect.sort_values(by="direct_effect", ascending=False).head(K)

    filtered = normalized_std_by_group[
        normalized_std_by_group["name"].isin(topK["name"])
    ]

    return filtered["direct_effect"].mean()


# %%
print(f"total : {std_by_group(big_dataframe)}")
# %%

for variable in ALL_NAR_VAR:
    direct_effect_df[variable]["narrative_variable"] = variable
    print(f"Variable: {variable}, {std_by_group(direct_effect_df[variable])}")
# %%


# %%
