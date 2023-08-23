from ..experiments import *
from ..plot_fn import *

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
variables = ["city", "character_name", "character_occupation", "season", "day_time"]
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
        querried_variables=[v for v in ALL_NAR_VAR if v != variable],
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
cor_var = "character_name"

df = []

remove_mean = True
for var in ALL_NAR_VAR[:3]:
    for cor_var in ALL_NAR_VAR[:3]:
        dla_ref = direct_logit_attribution(
            variable_datasets[var],
            model,
            caches_per_qvar[var],
            narrative_variable=var,
            ref_narrative_variable=None,
        )
        dla_cor = direct_logit_attribution(
            variable_datasets[var],
            model,
            rd_caches_per_qvar[cor_var],
            narrative_variable=var,
            ref_narrative_variable=None,
        )

        dla = dla_ref - dla_cor
        if remove_mean:
            dla = direct_logit_attribution(
                variable_datasets[var],
                model,
                caches_per_qvar[var],
                narrative_variable=var,
                ref_narrative_variable=None,
                remove_mean=True,
            )

        direct_path_patch = path_patching_logits(
            variable_datasets[var],
            model,
            caches_per_qvar[var],
            corrupted_cache=caches_per_qvar[cor_var],
            narrative_variable=var,
            ref_narrative_variable=None,
            probs=False,
            nb_ressamples=5,
            return_ref=False,
        )
        assert not isinstance(direct_path_patch, tuple)
        mean_dla = dla.mean(dim=-1)
        mean_direct_path_patch = direct_path_patch.mean(dim=-1)

        for l in range(mean_dla.shape[0]):
            for h in range(mean_dla.shape[1]):
                df.append(
                    {
                        "var": var,
                        "cor_var": cor_var,
                        "layer": l,
                        "head": h,
                        "dla": mean_dla[l, h].item(),
                        "direct_path_patch": mean_direct_path_patch[l, h].item(),
                    }
                )
# %%
df = pd.DataFrame(df)
# %%
fig = px.scatter(
    df,
    x="dla",
    y="direct_path_patch",
    hover_data=["layer", "head"],
    facet_col="var",
    facet_row="cor_var",
    title=f"Raw DLDA vs Direct Path Patching",
    height=1000,
    width=1000,
)

n_cols = len(df["var"].unique())
n_rows = len(df["cor_var"].unique())

# we add the line to each facet. Note that we start counting from 1
for i in range(1, n_rows * n_cols + 1):
    fig.add_shape(
        type="line",
        x0=df["dla"].min(),
        y0=df["dla"].min(),
        x1=df["dla"].max(),
        y1=df["dla"].max(),
        xref=f"x{i}",  # update according to facet
        yref=f"y{i}",  # update according to facet
        line=dict(color="Red", dash="dot"),
    )

fig.show()


# %% Evaluate the variance of direct path patching

all_direct_patch = []
var = "city"
cor_var = "city"
for i in range(10):
    all_direct_patch.append(
        path_patching_logits(
            variable_datasets[var],
            model,
            caches_per_qvar[var],
            corrupted_cache=rd_caches_per_qvar[cor_var],
            narrative_variable=var,
            ref_narrative_variable=None,
            probs=False,
            nb_ressamples=1,
        ).unsqueeze(  # type: ignore
            0
        )
    )

all_direct_patch = torch.cat(all_direct_patch, dim=0)

all_direct_patch_std = all_direct_patch.std(dim=0).mean(dim=-1)
all_direct_patch_mean = all_direct_patch.mean(dim=0).mean(dim=-1)

px.imshow(all_direct_patch_std.cpu().numpy())
fig = px.scatter(
    x=all_direct_patch_mean.cpu().flatten().numpy(),
    y=all_direct_patch_std.flatten().cpu().numpy(),
)
fig.update_layout(
    xaxis_title="Mean Direct Path Patching",
    yaxis_title="Std Direct Path Patching accross resampling",
    height=1000,
    width=1000,
)
# %% One single sample

var = "city"
cor_var = "city"
complement_datasets = {}

unique_sample_dataset = {}
for var in ALL_NAR_VAR:
    unique_sample_dataset[var] = NanoQADataset(
        name=var,
        nb_samples=1,
        nb_variable_values=5,
        querried_variables=[var],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )

caches_per_qvar_unique = {}
for variable in ALL_NAR_VAR:
    caches_per_qvar_unique[variable] = {}
    fill_cache(
        caches_per_qvar_unique[variable],
        model,
        unique_sample_dataset[variable],
        mid_layer=MID_LAYER,
    )
# %%
var = "character_occupation"
all_direct_patch_unique = []
for i in range(10):
    all_direct_patch_unique.append(
        path_patching_logits(
            unique_sample_dataset[var],
            model,
            caches_per_qvar_unique[var],
            cor_dataset=complement_datasets[cor_var],
            corrupted_cache=rd_caches_per_qvar[cor_var],
            narrative_variable=var,
            ref_narrative_variable=None,
            probs=False,
            nb_ressamples=1,
        ).unsqueeze(  # type: ignore
            0
        )
    )
# %%
all_direct_patch = torch.cat(all_direct_patch_unique, dim=0)

all_direct_patch_std = all_direct_patch.std(dim=0)
all_direct_patch_mean = all_direct_patch.mean(dim=0).mean(dim=-1)

# %%
import torch
import plotly.graph_objects as go

# ... [Your existing code]

# 1. Get indices of top 5% absolute values
num_elements = int(0.03 * all_direct_patch_mean.numel())
indices = torch.topk(torch.abs(all_direct_patch_mean).flatten(), num_elements)[1]

# Convert flattened indices to 2D indices
x_indices = indices // all_direct_patch_mean.shape[1]
y_indices = indices % all_direct_patch_mean.shape[1]

# Create labels based on conditions provided
labels = []
for x, y in zip(x_indices, y_indices):
    if y == 32:
        labels.append(f"MLP {x}")
    else:
        labels.append(f"H{x}L{y}")

# 2. Extract values and stds using the indices
selected_means = all_direct_patch_mean.flatten()[indices]
selected_stds = all_direct_patch_std.flatten()[indices]

# 3. Plot using Plotly
fig = go.Figure(
    data=[
        go.Bar(
            x=labels,
            y=selected_means,
            error_y=dict(type="data", array=selected_stds, visible=True),
            marker=dict(color=selected_means, colorscale="Viridis"),
        )
    ]
)

fig.update_layout(
    title="Top 5% Important Values with Error Bars",
    xaxis_title="Index",
    yaxis_title="Value",
)

fig.show()
# %%
# compare overlap accross the same variable

# %%
