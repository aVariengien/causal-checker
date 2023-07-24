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
nano_qa_dataset = NanoQADataset(
    name="general",
    nb_samples=30,
    nb_variable_values=5,
    querried_variables=ALL_NAR_VAR,  # ALL_NAR_VAR
    tokenizer=model.tokenizer,  # type: ignore
    seed=None,
)

# %%

d = evaluate_model(model, nano_qa_dataset, batch_size=100)
print_performance_table(d)
# %%

cache = {}
fill_cache(cache, model, nano_qa_dataset, mid_layer=MID_LAYER)


# %%

direct_path_patch, ref_ld = path_patching_logits(
    nano_qa_dataset,
    model,
    cache,
    corrupted_cache=cache,
    narrative_variable="correct",
    ref_narrative_variable=None,
    probs=False,
    nb_ressamples=1,
    return_ref=True,
)

# %%
print(ref_ld.mean())

# %%
_ = plot_sorted_direct_effect_bar_chart(
    dataset=nano_qa_dataset,
    model=model,
    cache=cache,
    variable="correct",
    alt_variable=None,
    title_suffix=f" - Target token Q_t(C_t) - {nano_qa_dataset.querried_variables}",
    error_bar=True,
    nb_ressamples=1,
)

# %%
# create datasets
DATASET_SIZE = 20
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


# %%


var_source = "character_name"
var_target = "season"
variable_dataset = variable_datasets


# %%
def pipeline_comparision(
    model: HookedTransformer,
    var_source: str,
    var_target: str,
    variable_dataset: Dict[str, NanoQADataset],
):
    # %%
    # request patching
    model.reset_hooks()

    source_dataset = variable_dataset[var_source]
    target_dataset = variable_dataset[var_target]
    baseline_dataset = NanoQADataset(
        name="baseline",
        nb_samples=DATASET_SIZE,
        nb_variable_values=5,
        querried_variables=[
            v for v in ALL_NAR_VAR if v not in [var_source, var_target]
        ],
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
    # %%
    # get movers data

    df_movers_t_data_t_token = get_mover_df(
        target_dataset,
        model,
        narrative_variable=var_target,
        mid_layer=MID_LAYER,
        cache=cache["target"],
        corrupted_cache=cache["baseline"],
    )
    df_movers_t_data_s_token = get_mover_df(
        target_dataset,
        model,
        narrative_variable=var_source,
        mid_layer=MID_LAYER,
        cache=cache["target"],
        corrupted_cache=cache["baseline"],
    )
    df_movers_s_data_s_token = get_mover_df(
        source_dataset,
        model,
        narrative_variable=var_source,
        mid_layer=MID_LAYER,
        cache=cache["source"],
        corrupted_cache=cache["baseline"],
    )

    # apply request patching

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

    # print the perfs
    fn = partial(
        apply_steering,
        target_dataset=target_dataset,
        model=model,
        cache_source=cache["source"],
        mid_layer=MID_LAYER,
    )

    print_perf_summary_patching(
        model=model,
        c1_q1=target_dataset,
        c2_q2=source_dataset,
        var1=var_target,
        var2=var_source,
        apply_hook_fn=fn,
    )

    # get movers post request patching
    df_movers_t_data_t_token_post_patching = get_mover_df(
        target_dataset,
        model,
        narrative_variable=var_target,
        mid_layer=MID_LAYER,
        cache=cache_post_patching["target"],
        corrupted_cache=cache["baseline"],
    )
    df_movers_t_data_s_token_post_patching = get_mover_df(
        target_dataset,
        model,
        narrative_variable=var_source,
        mid_layer=MID_LAYER,
        cache=cache_post_patching["target"],
        corrupted_cache=cache["baseline"],
    )

    # plot comparisons

    # source -> target

    # DLDA
    # On target data, target token, pre vs post patching
    # %%
    def plot_direct_effect_comparison(
        patching_cache: Dict, patching_type: str, percentile=0.98, error_bar=True
    ):
        _ = plot_comparison_direct_effect_bar_chart(
            dataset_1=target_dataset,
            cache_1=cache["target"],
            dataset_2=target_dataset,
            cache_2=patching_cache,
            model=model,
            variable=var_target,
            alt_variable=var_source,
            title_suffix=" - Target token Q_t(C_t)",
            error_bar=error_bar,
            height=800,
            width=1200,
            name_1=f"Pre {patching_type}",
            name_2=f"Post {patching_type}",
            percentile=percentile,
            corrupted_cache=cache["baseline"],
        )

        _ = plot_comparison_direct_effect_bar_chart(
            dataset_2=target_dataset,
            cache_2=cache["target"],
            dataset_1=target_dataset,
            cache_1=patching_cache,
            model=model,
            variable=var_source,
            alt_variable=var_target,
            title_suffix=" - Source token Q_s(C_t)",
            error_bar=error_bar,
            height=800,
            width=1200,
            name_2=f"Pre {patching_type}",
            name_1=f"Post {patching_type}",
            percentile=percentile,
            corrupted_cache=cache["baseline"],
        )

        df1, df2 = plot_comparison_direct_effect_bar_chart(
            dataset_1=target_dataset,
            cache_1=cache["chimera"],
            dataset_2=target_dataset,
            cache_2=patching_cache,
            model=model,
            variable=var_source,
            alt_variable=var_target,
            title_suffix=" - Source token Q_s(C_t)",
            error_bar=error_bar,
            height=800,
            width=1200,
            name_1="Baseline dataset 'C_t|Q_s'",
            name_2=f"Post {patching_type}",
            percentile=percentile,
            corrupted_cache=cache["baseline"],
        )
        return df1, df2

    # %%

    plot_direct_effect_comparison(
        cache_post_patching["target"],
        "Request-Patching",
        percentile=0.95,
        error_bar=False,
    )
    # movers plots

    # Target token target data pre/post patching

    plot_dataframe_comparison(
        df_movers_t_data_t_token,
        df_movers_t_data_t_token_post_patching,
        name1="Pre Request patching - Target token Q_t(C_t)",
        name2="Post Request patching - Target token Q_t(C_t)",
        variable_name=var_target,
    )

    # Target token target data post patching

    plot_dataframe_comparison(
        df_movers_t_data_s_token,
        df_movers_t_data_s_token_post_patching,
        name1="Pre Request patching - Source token Q_s(C_t)",
        name2="Post Request patching - Source token Q_s(C_t)",
        variable_name=var_source,
    )

    plot_dataframe_comparison(
        df_movers_s_data_s_token,
        df_movers_t_data_s_token_post_patching,
        name1="Baseline dataset 'C_t|Q_s' - Source token Q_s(C_t)",
        name2="Post Request patching - Source token Q_s(C_t)",
        variable_name=var_source,
    )

    # apply attention patching

    # movers = union_movers(
    #     dataset1=target_dataset,
    #     dataset2=source_dataset,
    #     df1=df_movers_t_data_t_token,
    #     df2=df_movers_s_data_s_token,
    #     percentile=0.0,
    #     filter_by_layer=False,
    # )
    # %%
    movers = [
        (i, j) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)
    ]

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
    # %%
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
    # %%
    df_attn, df_baseline = plot_direct_effect_comparison(
        cache_post_attn_patching, "Attn Ptrn Patching", error_bar=False, percentile=0.95
    )


# %%


# %%
plot_num_components(df_attn, np.linspace(0, 0.5, 20))
# %%
plot_num_components(df_baseline, np.linspace(0, 0.5, 20))
# %% scrap


# %%

direct_path_patch_attn, ref_ld, rd_ref = path_patching_logits(
    target_dataset,
    model,
    cache_post_attn_patching,
    corrupted_cache=rd_caches_per_qvar["city"],
    narrative_variable=var_source,
    ref_narrative_variable=None,
    probs=False,
    log_probs=True,
    nb_ressamples=1,
    return_ref=True,
)
print(
    f"Ref LD: {ref_ld.mean()}, Rand LD: {rd_ref.mean()}, Diff: {ref_ld.mean()-rd_ref.mean()}"
)


show_mtx(direct_path_patch_attn.mean(dim=-1).numpy(), title="Attn Ptrn Patching")

# %%

plt.hist(rd_ref, bins=20)

# %%
direct_path_patch_chimera, ref_ld, rd_ref = path_patching_logits(
    t_stories_s_questions,
    model,
    cache["chimera"],
    narrative_variable=var_source,
    corrupted_cache=rd_caches_per_qvar["city"],
    ref_narrative_variable=None,
    probs=False,
    log_probs=False,
    nb_ressamples=1,
    return_ref=True,
)
print(
    f"Ref LD: {ref_ld.mean()}, Rand LD: {rd_ref.mean()}, Diff: {ref_ld.mean()-rd_ref.mean()}"
)


show_mtx(direct_path_patch_chimera.mean(dim=-1).numpy(), title="Chimera baseline")

# %%

plt.hist(rd_ref, bins=20)

# %%
show_mtx(
    direct_path_patch_chimera.mean(dim=-1).numpy()
    - direct_path_patch_attn.mean(dim=-1).numpy(),
    title="Diff [chimera] - [Attn Ptrn Patching]",
)

# %% with DLA

dla_attn = direct_logit_attribution(
    target_dataset,
    model,
    cache_post_attn_patching,
    narrative_variable=var_source,
    ref_narrative_variable=var_target,
    remove_mean=True,
)

mean_dla_attn = dla_attn.mean(dim=-1).numpy()

show_mtx(mean_dla_attn, title="DLA attn patching")

# %%
dla_chimera = direct_logit_attribution(
    t_stories_s_questions,
    model,
    cache["chimera"],
    narrative_variable=var_source,
    ref_narrative_variable=var_target,
    remove_mean=True,
)
mean_dla_chimera = dla_chimera.mean(dim=-1).numpy()
show_mtx(mean_dla_chimera, title="DLA Chimera baseline")

# %%
show_mtx(
    dla_chimera.mean(dim=-1).numpy() - dla_attn.mean(dim=-1).numpy(),
    title="Diff DLA [chimera] - [Attn Ptrn Patching]",
)
# %%

# let's look at the logits


last_resid = cache_post_attn_patching["blocks.31.hook_resid_post-out-END"]
last_resid = cache["chimera"]["blocks.31.hook_resid_post-out-END"]


def model_end(x):
    """Compute the final layer norm and unembedding"""
    res = model.ln_final(x.cuda().unsqueeze(0))
    return model.unembed(res).cpu().squeeze(0)


logits = model_end(last_resid)

# %%


def print_logit_value(toks: List[str], story_idx, logits, prob=False):
    if prob:
        logits = torch.softmax(logits, dim=-1)
    logit_vals = []
    for tok in toks:
        tok_idx = model.tokenizer(tok, return_tensors="pt")["input_ids"].squeeze(0)  # type: ignore
        if len(tok_idx) > 1:
            raise ValueError(f"Token {tok} is not a single token")
        logit_vals.append(logits[story_idx, tok_idx].item())
    print(f"Logit values for story {story_idx} and tokens {toks}:")
    for tok, val in zip(toks, logit_vals):
        if val < 0.01:
            print(f"{tok}: {val:.2e}")
        else:
            print(f"{tok}: {val:.2f}")
    print(f"Average: {np.mean(logit_vals):.2f}")


# %%
pprint_nanoqa_prompt(target_dataset, 1)
# %%

print_logit_value([' "', " Bob"], 1, logits, prob=False)
# %%
print_logit_value(
    [" Jessica", " morning", " summer", " Bus", " astr"], 0, logits, prob=True
)
# %%
print_logit_value(
    [" libr", " after", " C", " winter", " Michael"], 0, logits, prob=True
)

# %%
print_logit_value(
    [" winter", " fall", " spring", " summer", " Michael"], 0, logits, prob=True
)

# %%

print_logit_value(
    [" Bob", " Matthew", " Ashley", " Jessica", " Michael"], 0, logits, prob=True
)

# %%
print_logit_value([" Matthew", " evening", " fall", " Port", " architect"], 5, logits)
# %%
pprint_nanoqa_prompt(target_dataset, 5)
# %%
print_logit_value(
    [" evening", " spring", " Valencia", " Michael", " architect"], 5, logits
)
# %%
print_logit_value(
    [" Matthew", " Jessica", " Christopher", " Ashley", " Michael", " Bob"], 5, logits
)
# %%
