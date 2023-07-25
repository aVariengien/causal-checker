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

# %% Create the datasets
DATASET_SIZE = 40
variable_datasets = {}
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


var_source = "character_name"
var_target = "season"
variable_dataset = variable_datasets


# %% Setup the caches
model.reset_hooks()

source_dataset = variable_dataset[var_source]
target_dataset = variable_dataset[var_target]

baseline_vars = [v for v in ALL_NAR_VAR if v not in [var_source, var_target]]
baseline_dataset = NanoQADataset(
    name="baseline",
    nb_samples=DATASET_SIZE,
    nb_variable_values=5,
    querried_variables=baseline_vars,
    tokenizer=model.tokenizer,  # type: ignore
    seed=None,
)

t_stories_s_questions = target_dataset.question_from(source_dataset)
t_stories_s_questions.name = "t_stories_s_questions"

# compute cache

cache = {}
for dataset, name in zip(
    [source_dataset, target_dataset, t_stories_s_questions, baseline_dataset],
    ["source", "target", "chimera", "baseline"],
):
    cache[name] = {}
    fill_cache(cache[name], model, dataset, mid_layer=MID_LAYER)

# %% request patching
request_patch_fn = partial(
    apply_steering,
    target_dataset=target_dataset,
    model=model,
    cache_source=cache["source"],
    mid_layer=MID_LAYER,
)
apply_steering(
    target_dataset=target_dataset,
    model=model,
    cache_source=cache["source"],
    mid_layer=MID_LAYER,
)

# gather cache after request patching
cache_post_patching = {}
for dataset, name in zip(
    [target_dataset],
    ["target"],
):
    cache_post_patching[name] = {}
    fill_cache(cache_post_patching[name], model, dataset, mid_layer=MID_LAYER)

model.reset_hooks()
# %% print perf of request patching
print("==== Request Patching ====")
print_perf_summary_patching(
    model=model,
    c1_q1=target_dataset,
    c2_q2=source_dataset,
    var1=var_target,
    var2=var_source,
    apply_hook_fn=request_patch_fn,
)


# %% Attention patching
movers = [(i, j) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

# print the perfs

model.reset_hooks()
attn_patching_fn = partial(
    apply_attention_patch,
    model=model,
    movers=movers,
    source_cache=cache["chimera"],
    target_dataset=target_dataset,
    source_dataset=t_stories_s_questions,
    mode="all_seq",
)

print("==== Attention Pattern Patching ====")
print_perf_summary_patching(
    model=model,
    c1_q1=target_dataset,
    c2_q2=source_dataset,
    var1=var_target,
    var2=var_source,
    apply_hook_fn=attn_patching_fn,
)

# apply attention patching
attn_patching_fn()
cache_post_attn_patching = {}
fill_cache(cache_post_attn_patching, model, target_dataset, mid_layer=MID_LAYER)
model.reset_hooks()
# plot difference with non-patching baselines


# %%  #### PLOTTING ####


# %% attention heads plotting

df_movers_t_data_t_token = get_mover_df(
    target_dataset,
    model,
    narrative_variable=var_target,
    mid_layer=MID_LAYER,
    cache=cache["target"],
    corrupted_cache=cache["baseline"],
    ref_narrative_variable=baseline_vars,
)

df_movers_t_data_s_token = get_mover_df(
    target_dataset,
    model,
    narrative_variable=var_source,
    mid_layer=MID_LAYER,
    cache=cache["target"],
    corrupted_cache=cache["baseline"],
    ref_narrative_variable=baseline_vars,
)
df_movers_s_data_s_token = get_mover_df(
    source_dataset,
    model,
    narrative_variable=var_source,
    mid_layer=MID_LAYER,
    cache=cache["source"],
    corrupted_cache=cache["baseline"],
    ref_narrative_variable=baseline_vars,
)

df_movers_t_data_t_token_post_patching = get_mover_df(
    target_dataset,
    model,
    narrative_variable=var_target,
    mid_layer=MID_LAYER,
    cache=cache_post_patching["target"],
    corrupted_cache=cache["baseline"],
    ref_narrative_variable=baseline_vars,
)
df_movers_t_data_s_token_post_patching = get_mover_df(
    target_dataset,
    model,
    narrative_variable=var_source,
    mid_layer=MID_LAYER,
    cache=cache_post_patching["target"],
    corrupted_cache=cache["baseline"],
    ref_narrative_variable=baseline_vars,
)
# %%
plot_dataframe_comparison(
    df_movers_t_data_t_token,
    df_movers_t_data_t_token_post_patching,
    name1="Pre Request patching - Target token Q_t(C_t)",
    name2="Post Request patching - Target token Q_t(C_t)",
    variable_name=var_target,
    error_bars=False,
)

# Target token target data post patching

plot_dataframe_comparison(
    df_movers_t_data_s_token_post_patching,
    df_movers_t_data_s_token,
    name2="Pre Request patching - Source token Q_s(C_t)",
    name1="Post Request patching - Source token Q_s(C_t)",
    variable_name=var_source,
    error_bars=False,
)

plot_dataframe_comparison(
    df_movers_s_data_s_token,
    df_movers_t_data_s_token_post_patching,
    name1="Baseline dataset 'C_t|Q_s' - Source token Q_s(C_t)",
    name2="Post Request patching - Source token Q_s(C_t)",
    variable_name=var_source,
    error_bars=False,
)

# %% [markdown]
# # Direct effect plotting

# %% Direct effect plotting

df_baseline, df_attn = plot_comparison_direct_effect_bar_chart(
    dataset_1=target_dataset,
    cache_1=cache["chimera"],
    dataset_2=target_dataset,
    cache_2=cache_post_attn_patching,
    model=model,
    variable=var_source,
    alt_variable=baseline_vars,
    title_suffix=" - Source token Q_s(C_t)",
    error_bar=False,
    height=800,
    width=1200,
    name_1="Baseline dataset 'C_t|Q_s'",
    name_2=f"Post Attn pattern patching",
    percentile=0.95,
    corrupted_cache=cache["baseline"],
)

print("Relative Variation in direct effect == ATTN PATCHING")
print(compute_comparision_metric(df_attn, df_baseline))

#   compute overlap

df_post_patch, _ = compute_direct_effect_df(
    dataset=target_dataset,
    model=model,
    cache=cache_post_attn_patching,
    narrative_variable=var_source,
    ref_narrative_variable=None,
    corrupted_cache=cache["baseline"],
    nb_ressamples=3,
    expand_sample_dim=True,
)
df_post_patch["narrative_variable"] = var_source

df_pre_patch, _ = compute_direct_effect_df(
    dataset=t_stories_s_questions,
    model=model,
    cache=cache["chimera"],
    narrative_variable=var_source,
    ref_narrative_variable=None,
    corrupted_cache=cache["baseline"],
    nb_ressamples=3,
    expand_sample_dim=True,
)
df_pre_patch["narrative_variable"] = var_source

mean_overlap, std_overlap = comparative_overlap(
    df_pre_patch, df_post_patch, threshold=0.025
)
print(f"Mean overlap ATTN PATTERN PATCHING: {mean_overlap:.2f} +- {std_overlap:.2f}")


# %%
df_baseline, df_request = plot_comparison_direct_effect_bar_chart(
    dataset_1=target_dataset,
    cache_1=cache["chimera"],
    dataset_2=target_dataset,
    cache_2=cache_post_patching["target"],
    model=model,
    variable=var_source,
    alt_variable=baseline_vars,
    title_suffix=" - Source token Q_s(C_t)",
    error_bar=False,
    height=800,
    width=1200,
    name_1="Baseline dataset 'C_t|Q_s'",
    name_2=f"Post Request patching",
    percentile=0.95,
    corrupted_cache=cache["baseline"],
)

print("Relative Variation in direct effect == REQUEST PATCHING")
print(compute_comparision_metric(df_request, df_baseline))


df_post_patch, _ = compute_direct_effect_df(
    dataset=target_dataset,
    model=model,
    cache=cache_post_patching["target"],
    narrative_variable=var_source,
    ref_narrative_variable=baseline_vars,
    corrupted_cache=cache["baseline"],
    nb_ressamples=3,
    expand_sample_dim=True,
)
df_post_patch["narrative_variable"] = var_source

df_pre_patch, _ = compute_direct_effect_df(
    dataset=t_stories_s_questions,
    model=model,
    cache=cache["chimera"],
    narrative_variable=var_source,
    ref_narrative_variable=baseline_vars,
    corrupted_cache=cache["baseline"],
    nb_ressamples=3,
    expand_sample_dim=True,
)
df_pre_patch["narrative_variable"] = var_source


mean_overlap, std_overlap = comparative_overlap(
    df_pre_patch, df_post_patch, threshold=0.025
)
print(f"Mean overlap REQUEST PATCHING: {mean_overlap:.2f} +- {std_overlap:.2f}")

# %% Variation in attention probability after request patching
df1, df2 = df_movers_s_data_s_token, df_movers_t_data_s_token_post_patching
threshold_attn = 0.1
threshold_de = 0.01
filtering = (
    (df1["abs_normalized_direct_effect_mean"] > threshold_de)
    & (df1["attn_mean"] > threshold_attn)
) | (
    (df2["abs_normalized_direct_effect_mean"] > threshold_de)
    & (df2["attn_mean"] > threshold_attn)
)

df1_filtered = df1[filtering]
df2_filtered = df2[filtering]

df2.loc[filtering, "relative_variation"] = (
    df2_filtered["attn_mean"] - df1_filtered["attn_mean"]
) / df1_filtered["attn_mean"]

df2_filtered = df2[filtering]
print(
    f"Median relative variation in attn prob: {df2_filtered['relative_variation'].median()} +/- {df2_filtered['relative_variation'].std()}"
)

# %%
# USE RAW MATRICES

for patching_name, patching_cache in zip(
    ["Request patching D2->D1", "Attn pattching D3_attn->D1"],
    [cache_post_patching["target"], cache_post_attn_patching],
):
    print("\n" * 3)
    print(" ==== ", patching_name, " ====")
    direct_path_patch, ref_ld_p, rd_ref_p = path_patching_logits(
        target_dataset,
        model,
        cache=patching_cache,
        corrupted_cache=cache["baseline"],
        narrative_variable=var_source,
        ref_narrative_variable=baseline_vars,
        probs=False,
        nb_ressamples=1,
        return_ref=True,
    )
    print(
        f"Ref LD: {ref_ld_p.mean()}, Rand LD: {rd_ref_p.mean()}, Diff: {ref_ld_p.mean()-rd_ref_p.mean()}"
    )

    ref_ld_p, rd_ref_p = (
        ref_ld_p.numpy().mean(),
        rd_ref_p.numpy().mean(),
    )
    mean_de = direct_path_patch.mean(dim=-1).numpy()
    show_mtx(
        mean_de,
        model,
        title=f"Direct effect {patching_name}",
        color_map_label="Direct effect",
    )

    direct_path_patch_chimera, ref_ld, rd_ref = path_patching_logits(
        t_stories_s_questions,
        model,
        cache["chimera"],
        narrative_variable=var_source,
        corrupted_cache=cache["baseline"],
        ref_narrative_variable=baseline_vars,
        nb_ressamples=1,
        return_ref=True,
    )
    print(
        f"Ref LD: {ref_ld.mean()}, Rand LD: {rd_ref.mean()}, Diff: {ref_ld.mean()-rd_ref.mean()}"
    )

    mean_baseline_de = direct_path_patch_chimera.mean(dim=-1).numpy()

    show_mtx(
        mean_baseline_de,
        model,
        title="Baseline chimera",
        color_map_label="Direct effect",
    )

    rd_ref, ref_ld = rd_ref.numpy().mean(), ref_ld.numpy().mean()

    show_mtx(
        mean_baseline_de - mean_de,
        model,
        title=f"Diff [chimera] - [{patching_name}]",
        color_map_label="Direct effect variation",
    )

    diff = (mean_de - mean_baseline_de) / (ref_ld - rd_ref)
    show_mtx(
        diff,
        model,
        title=f"Relative direct effect variation, D3 vs {patching_name}",
        color_map_label="Relative direct effect variation",
    )

    print(
        f"Relative direct effect variation {patching_name}; Heads: {diff[:, :32].sum()}, Mlp: {diff[:, 32].sum()}"
    )


# %%


# %%
