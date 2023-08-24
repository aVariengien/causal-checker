if __name__ == "__main__":
    from .experiments import *
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
import plotly.graph_objects as go
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


torch.set_grad_enabled(False)


# %% plot functions
def plot_movers(
    dataset: NanoQADataset,
    model: Any,
    model_name: str,
    title: str = "",
    df: Optional[pd.DataFrame] = None,
    mid_layer=-1,
):
    if df is None:
        df = get_mover_df(dataset, model, mid_layer=mid_layer)
    assert df is not None
    # plots the results with error bars

    df_plot = df[
        (df["abs_direct_effect_mean"] > 0.01) & (df["attn_mean"] > 0.01)
    ].copy()

    fig = px.scatter(
        df_plot,
        x="abs_direct_effect_mean",
        y="attn_mean",
        error_x="direct_effect_std",
        error_y="attn_std",
        color="layer",
        hover_name="head_name",
        size=[4] * len(df_plot),
    )

    fig.update_layout(
        xaxis_title="Absolute direct Logit Attribution",
        yaxis_title="Attention to correct token",
        title=f"Absolute direct Logit Attribution vs Attention to correct token - {model_name} - {dataset.name} {title}",
        height=800,
        width=1200,
    )
    return fig


# %%
def expand_and_compute(df):
    # Expand df_xp
    df_xp_expanded = df.explode("direct_effect_all")
    df_xp_expanded["direct_effect"] = df_xp_expanded["direct_effect_all"].astype(float)
    df_xp_expanded["sample_idx"] = df_xp_expanded.groupby(["layer", "name"]).cumcount()
    return df_xp_expanded


def compute_comparision_metric(df_xp, df_baseline, abs=False):
    d_eff_baseline = float((df_baseline["ref_ld"] - df_baseline["cor_ld"]).iloc[0])
    # Step 1: Extract component type from the 'name' column

    df_xp_expanded = expand_and_compute(df_xp)
    df_baseline_expanded = expand_and_compute(df_baseline)

    df_xp_expanded["component_type"] = df_baseline_expanded["name"].apply(
        lambda x: x.split()[0]
    )
    df_baseline_expanded["component_type"] = df_baseline_expanded["name"].apply(
        lambda x: x.split()[0]
    )

    # Step 2: Compute the difference between direct_effect_mean
    df_xp_expanded["effect_ratio"] = (
        df_xp_expanded["direct_effect"] - df_baseline_expanded["direct_effect"]
    ) / d_eff_baseline

    # Step 5: Sum up the results for each component type
    df_summed_effect_ratio = (
        df_xp_expanded.groupby(["component_type", "sample_idx"])["effect_ratio"]
        .sum()
        .reset_index()
    )
    df_mean_std_effect_ratio = (
        df_summed_effect_ratio.groupby("component_type")["effect_ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )

    return df_mean_std_effect_ratio


# %%


def calculate_components(df, proportion=0.8):
    df_sorted = df.sort_values("abs_direct_effect_mean", ascending=False)
    df_sorted["abs_direct_effect_mean_pct"] = (
        df_sorted["abs_direct_effect_mean"] / df_sorted["abs_direct_effect_mean"].sum()
    )
    df_sorted["cumulative_pct"] = df_sorted["abs_direct_effect_mean_pct"].cumsum()
    num_components = len(df_sorted[df_sorted["cumulative_pct"] <= proportion])
    return num_components


def plot_num_components(df, proportions):
    df_sorted = df.sort_values("abs_direct_effect_mean", ascending=False)
    df_sorted["abs_direct_effect_mean_pct"] = (
        df_sorted["abs_direct_effect_mean"] / df_sorted["abs_direct_effect_mean"].sum()
    )
    df_sorted["cumulative_pct"] = df_sorted["abs_direct_effect_mean_pct"].cumsum()
    num_components = []

    for proportion in proportions:
        components = len(df_sorted[df_sorted["cumulative_pct"] <= proportion])
        num_components.append(components)

    plt.plot(proportions, num_components, marker="o")
    plt.xlabel("Proportion of Cumulative abs_direct_effect_mean")
    plt.ylabel("Number of Components")
    plt.title(
        "Number of Components for Different Proportions of Cumulative abs_direct_effect_mean"
    )
    plt.show()


def plot_dataframe_comparison(
    df1,
    df2,
    name1="df1",
    name2="df2",
    variable_name="",
    marker_size=15,
    error_bars=True,
    save_fig=False,
):
    """Plot a comparison of two dataframes with the same columns, df1 is the reference dataframe"""
    ref_total_effect = df1["total_effect"].iloc[0].numpy()
    df1["abs_normalized_direct_effect_mean"] = (
        df1["direct_effect_mean"].abs() / ref_total_effect
    )
    df2["abs_normalized_direct_effect_mean"] = (
        df2["direct_effect_mean"].abs() / ref_total_effect
    )

    df1["normalized_direct_effect_std"] = df1["direct_effect_std"] / ref_total_effect
    df2["normalized_direct_effect_std"] = df2["direct_effect_std"] / ref_total_effect

    # Filter points based on condition
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

    fig = go.Figure()
    # Plot lines connecting corresponding points
    for i in range(len(df1_filtered)):
        fig.add_trace(
            go.Scatter(
                x=[
                    df1_filtered["abs_normalized_direct_effect_mean"].iloc[i],
                    df2_filtered["abs_normalized_direct_effect_mean"].iloc[i],
                ],
                y=[
                    df1_filtered["attn_mean"].iloc[i],
                    df2_filtered["attn_mean"].iloc[i],
                ],
                mode="lines",
                line=dict(color="rgba(0, 0, 0, 0.1)"),
                showlegend=False,
            )
        )
    # Plot points for df1 with triangles
    fig.add_trace(
        go.Scatter(
            x=df1_filtered["abs_normalized_direct_effect_mean"],
            y=df1_filtered["attn_mean"],
            mode="markers",
            name=name1,
            marker=dict(
                symbol="triangle-up",
                size=marker_size,
                color=df1_filtered["layer"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Layer"),  # Add color bar here
            ),
            error_x=dict(
                type="data",
                array=df1_filtered["normalized_direct_effect_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            )
            if error_bars
            else None,
            error_y=dict(
                type="data",
                array=df1_filtered["attn_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            )
            if error_bars
            else None,
        )
    )

    # Plot points for df2 with circles
    fig.add_trace(
        go.Scatter(
            x=df2_filtered["abs_normalized_direct_effect_mean"],
            y=df2_filtered["attn_mean"],
            mode="markers",
            name=name2,
            marker=dict(
                symbol="circle",
                size=marker_size,
                color=df2_filtered["layer"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Layer"),  # Add color bar here
            ),
            error_x=dict(
                type="data",
                array=df2_filtered["normalized_direct_effect_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            )
            if error_bars
            else None,
            error_y=dict(
                type="data",
                array=df2_filtered["attn_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            )
            if error_bars
            else None,
        )
    )

    fig.update_layout(
        xaxis_title="Normalized Direct Effect",
        yaxis_title="Attention to first token",
        title=f"Absolute Direct Logit Attribution vs Attention to correct token (variable: {variable_name})",
        height=800,
        width=1200,
        # legend=dict(x=100, y=1000),  # Adjust legend position here
    )
    fig.show()
    if save_fig:
        fig.write_image(f"figs/fig_2d_heads_{name1}_{name2}.pdf")


def plot_intersection_mtx(d: Dict[str, List], title: str = ""):
    """Plot intersection matrix of the given dictionary of lists"""
    intersec_mtx = np.zeros((len(d), len(d)))
    for i, (k1, v1) in enumerate(d.items()):
        for j, (k2, v2) in enumerate(d.items()):
            intersec_mtx[i, j] = len(set(v1).intersection(set(v2)))

    return px.imshow(intersec_mtx, x=list(d.keys()), y=list(d.keys()), title=title)


def compute_direct_effect_df(
    dataset: NanoQADataset,
    model: HookedTransformer,
    cache: Dict,
    narrative_variable: str,
    ref_narrative_variable: Optional[Union[str, List[str]]],
    corrupted_cache: Optional[Dict] = None,
    nb_ressamples=1,
    expand_sample_dim=False,
):
    direct_effect, ref_ld, cor_ld = path_patching_logits(
        dataset,
        model,
        cache,
        narrative_variable=narrative_variable,
        ref_narrative_variable=ref_narrative_variable,
        nb_ressamples=nb_ressamples,
        corrupted_cache=corrupted_cache,
        return_ref=True,
    )
    direct_effect_mean = direct_effect.mean(dim=-1)
    direct_effect_std = direct_effect.std(dim=-1)
    df = []

    total_abs_direct_effect = ref_ld.mean() - cor_ld.mean()

    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads + 1):
            if h == model.cfg.n_heads:
                name = f"mlp {l}"
            else:
                name = f"head {l} {h}"
            df.append(
                {
                    "layer": l,
                    "name": name,
                    "direct_effect_mean": direct_effect_mean[l, h].numpy(),
                    "abs_direct_effect_mean": direct_effect_mean[l, h].abs().numpy(),
                    "direct_effect_std": direct_effect_std[l, h].numpy(),
                    "direct_effect_all": direct_effect[l, h].numpy(),
                    "ref_ld": ref_ld.mean(),
                    "cor_ld": cor_ld.mean(),
                    "normalized_direct_effect_mean": direct_effect_mean[l, h].numpy()
                    / total_abs_direct_effect,
                    "normalized_direct_effect_std": direct_effect_std[l, h].numpy()
                    / total_abs_direct_effect,
                    "all_normalized_direct_effect": direct_effect[l, h].numpy()
                    / total_abs_direct_effect,
                }
            )
    df = pd.DataFrame(df)
    if expand_sample_dim:
        df = expand_and_compute(df)
    return df, total_abs_direct_effect


def plot_sorted_direct_effect_bar_chart(
    dataset: NanoQADataset,
    model: HookedTransformer,
    cache: Dict,
    variable: str,
    alt_variable: Optional[str] = None,
    title_suffix="",
    error_bar=True,
    height=800,
    width=1200,
    nb_ressamples=1,
    df=None,
):
    if df is None:
        df, total_abs_direct_effect = compute_direct_effect_df(
            dataset,
            model,
            cache,
            narrative_variable=variable,
            ref_narrative_variable=alt_variable,
            nb_ressamples=nb_ressamples,
        )
    else:
        total_abs_direct_effect = (df["ref_ld"] - df["cor_ld"]).iloc[0]

    df_sorted = df.sort_values("abs_direct_effect_mean", ascending=False)
    threshold = df_sorted["abs_direct_effect_mean"].quantile(0.95)
    df_filtered = df_sorted[df_sorted["abs_direct_effect_mean"] > threshold]

    x = df_filtered["name"]
    y = df_filtered["direct_effect_mean"]

    fig = go.Figure()

    if error_bar:
        error = df_filtered["direct_effect_std"]
        fig.add_trace(
            go.Bar(
                x=x, y=y, error_y=dict(type="data", array=error), name="Mean with Error"
            )
        )
    else:
        fig.add_trace(go.Bar(x=x, y=y, name="Mean"))

    fig.update_layout(
        title=f"Top 5% of Largest abs_direct_effect_mean Entries (Total {total_abs_direct_effect:.2f})"
        + title_suffix,
        xaxis=dict(title="Name"),
        yaxis=dict(title="direct_effect_mean"),
        xaxis_tickangle=-90,
        width=width,
        height=height,
    )

    fig.show()

    return df


def plot_comparison_direct_effect_bar_chart(
    dataset_1: NanoQADataset,
    cache_1: Dict,
    dataset_2: NanoQADataset,
    cache_2: Dict,
    model: HookedTransformer,
    variable: str,
    corrupted_cache: Optional[Dict] = None,
    alt_variable: Optional[Union[str, List[str]]] = None,
    title_suffix="",
    error_bar=True,
    height=800,
    width=1200,
    percentile=0.95,
    name_1="Dataset1",
    name_2="Dataset2",
    save_fig=False,
):
    # TODO probably correct
    # Compute direct_effect for the 1 dataset
    df_1, total_direct_effect1 = compute_direct_effect_df(
        dataset_1,
        model,
        cache_1,
        narrative_variable=variable,
        ref_narrative_variable=alt_variable,
        corrupted_cache=corrupted_cache,
    )

    # Compute direct_effect for the 2 dataset
    df_2, total_direct_effect2 = compute_direct_effect_df(
        dataset_2,
        model,
        cache_2,
        narrative_variable=variable,
        ref_narrative_variable=alt_variable,
        corrupted_cache=corrupted_cache,
    )

    name_1 = f"{name_1} (Total abs dir. effect {total_direct_effect1:.2f})"
    name_2 = f"{name_2} (Total abs dir. effect {total_direct_effect2:.2f})"

    # Sort bars based on 1 dataset's values
    df_sorted = df_1.sort_values("abs_direct_effect_mean", ascending=False)
    sorting_order = df_sorted.index

    df_2 = df_2.loc[sorting_order]
    df_1 = df_1.loc[sorting_order]

    threshold = df_sorted["abs_direct_effect_mean"].quantile(percentile)
    df_filtered = df_sorted[df_sorted["abs_direct_effect_mean"] > threshold]

    # Create grouped bars for 2 and 1 direct_effect
    x = df_filtered["name"]

    y_2 = df_2[df_2["name"].isin(x)]["normalized_direct_effect_mean"]
    y_1 = df_1[df_1["name"].isin(x)]["normalized_direct_effect_mean"]

    fig = go.Figure()

    if error_bar:
        error_2 = df_2[df_2["name"].isin(df_filtered["name"])][
            "normalized_direct_effect_std"
        ]
        error_1 = df_1[df_1["name"].isin(df_filtered["name"])][
            "normalized_direct_effect_std"
        ]

        fig.add_trace(
            go.Bar(
                x=x,
                y=y_2,
                error_y=dict(type="data", array=error_2),
                name=name_2,
                marker_color="blue",  # Color for 2 bars
            )
        )

        fig.add_trace(
            go.Bar(
                x=x,
                y=y_1,
                error_y=dict(type="data", array=error_1),
                name=name_1,
                marker_color="red",  # Color for 1 bars
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=x,
                y=y_2,
                name=name_2,
                marker_color="blue",  # Color for 2 bars
            )
        )

        fig.add_trace(
            go.Bar(
                x=x,
                y=y_1,
                name=name_1,
                marker_color="red",  # Color for 1 bars
            )
        )

    fig.update_layout(
        title=f"Top {int(100-100*percentile)}% of Largest abs_direct_effect_mean Entries - {variable} -"
        + title_suffix,
        xaxis=dict(title="Name"),
        yaxis=dict(title="Normalized direct effect"),
        xaxis_tickangle=-90,
        width=width,
        height=height,
        barmode="group",  # Grouped bars
    )

    fig.show()
    if save_fig:
        fig.write_image(
            f"figs/fig_bar_chart_comparison_{name_1}_{name_2}_{variable}_{title_suffix}.pdf"
        )
    return df_1, df_2


def print_perf_summary_patching(
    model: HookedTransformer,
    c1_q1: NanoQADataset,
    c2_q2: NanoQADataset,
    var1: str,
    var2: str,
    apply_hook_fn: Callable,
):
    model.reset_hooks()

    c1_q2 = c1_q1.question_from(c2_q2)

    labels = [c1_q1, c2_q2, c1_q2]
    logits = {}

    for dataset, name in zip([c1_q1, c2_q2, c1_q2], ["c1_q1", "c2_q2", "c1_q2"]):
        logits[name] = model(dataset.prompts_tok).detach().cpu()

    model.reset_hooks()
    apply_hook_fn()
    logits["c1_q1 post-patching"] = model(c1_q1.prompts_tok)
    model.reset_hooks()

    perfs = {}
    for dataset, name in zip(
        [c1_q1, c2_q2, c1_q2, c1_q1], ["c1_q1", "c2_q2", "c1_q2", "c1_q1 post-patching"]
    ):
        perfs[name] = {}
        for label_dataset, label_name in zip(labels, ["c1_q1", "c2_q2", "c1_q2"]):
            perfs[name][label_name] = evaluate_model(
                model,
                dataset,
                batch_size=100,
                logits=logits[name],
                label_nano_qa_dataset=label_dataset,
            )

    print("Performances")
    for name in ["c1_q1", "c2_q2", "c1_q2", "c1_q1 post-patching"]:
        s = f"= Dataset: {name} = "
        for label_name, var_name in zip(
            ["c1_q1", "c2_q2", "c1_q2"], [var1, var2, var2]
        ):
            s += f"{label_name} ({var_name}): {perfs[name][label_name][var_name+'_prob_mean']:.2f}± {perfs[name][label_name][var_name+'_prob_std']:.2f} | "
        print(s)


def union_movers(
    dataset1: NanoQADataset,
    df1: pd.DataFrame,
    dataset2: NanoQADataset,
    df2: pd.DataFrame,
    model: HookedTransformer,
    MID_LAYER: int,
    percentile: float = 0.0,
    filter_by_layer: bool = False,
):
    movers1 = get_movers_names(
        dataset1,
        model,
        percentile=percentile,
        df=df1,
        filter_by_layer=filter_by_layer,
        mid_layer=MID_LAYER,
    )
    movers2 = get_movers_names(
        dataset2,
        model,
        percentile=percentile,
        df=df2,
        filter_by_layer=filter_by_layer,
        mid_layer=MID_LAYER,
    )
    all_movers = movers2
    for m in movers1:
        if not m in all_movers:
            all_movers.append(m)
    return all_movers


def show_mtx(
    mtx,
    model: HookedTransformer,
    title="NO TITLE :(",
    color_map_label="Logit diff variation",
    return_fig=False,
):
    """Show a plotly matrix with a centered color map. Designed to display results of path patching experiments."""
    # we center the color scale on zero by defining the range (-max_abs, +max_abs)
    max_val = float(max(abs(mtx.min()), abs(mtx.max())))
    x_labels = [f"h{i}" for i in range(model.cfg.n_layers)] + ["mlp"]
    fig = px.imshow(
        mtx,
        title=title,
        labels=dict(x="Head", y="Layer", color=color_map_label),
        color_continuous_scale="RdBu",
        range_color=(-max_val, max_val),
        x=x_labels,
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
    )
    if return_fig:
        return fig
    else:
        fig.show()


def get_mover_df(
    dataset: NanoQADataset,
    model: Any,
    narrative_variable: str = "correct",
    mid_layer=-1,
    cache: Optional[Dict] = None,
    corrupted_cache: Optional[Dict] = None,
    ref_narrative_variable: Optional[Union[str, List[str]]] = None,
):
    if cache is None:
        cache = {}
        fill_cache(cache, model, dataset, mid_layer=mid_layer)
    direct_effect_df, _ = compute_direct_effect_df(
        dataset=dataset,
        model=model,
        cache=cache,
        narrative_variable=narrative_variable,
        ref_narrative_variable=ref_narrative_variable,
        corrupted_cache=corrupted_cache,
        nb_ressamples=3,
    )
    attn_to_cor_tok = attention_to_nar_var_tok(
        dataset, model, cache, narrative_variable=narrative_variable
    )
    df = []

    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            head_name = f"head {l} {h}"
            head_df = direct_effect_df[direct_effect_df["name"] == head_name]

            df.append(
                {
                    "layer": l,
                    "head": h,
                    "direct_effect_mean": head_df["direct_effect_mean"].iloc[0],
                    "abs_normalized_direct_effect_mean": head_df[
                        "normalized_direct_effect_mean"
                    ]
                    .iloc[0]
                    .abs(),
                    "direct_effect_std": head_df["direct_effect_std"].iloc[0]
                    / math.sqrt(len(dataset)),
                    "attn_mean": attn_to_cor_tok[l, h].mean(),
                    "attn_std": attn_to_cor_tok[l, h].std() / math.sqrt(len(dataset)),
                    "head_name": f"l {l} h {h}",
                    "total_effect": head_df["ref_ld"].iloc[0]
                    - head_df["cor_ld"].iloc[0],
                }
            )
    df = pd.DataFrame(df)
    return df


def get_movers_names(
    dataset: NanoQADataset,
    model: Any,
    percentile: float = 0.9,
    df: Optional[pd.DataFrame] = None,
    narrative_variable: str = "correct",
    mid_layer: int = -1,
    filter_by_layer: bool = False,
):
    if df is None:
        df = get_mover_df(
            dataset, model, narrative_variable=narrative_variable, mid_layer=mid_layer
        )
    attn_threshold = df["attn_mean"].quantile(percentile)
    direct_effect_threshold = df["abs_normalized_direct_effect_mean"].quantile(
        percentile
    )

    filtering = (df["abs_normalized_direct_effect_mean"] > direct_effect_threshold) & (
        df["attn_mean"] > attn_threshold
    )
    if filter_by_layer:
        filtering = filtering & (df["layer"] >= mid_layer)
    movers = df[filtering]
    return list(movers[["layer", "head"]].to_records(index=False))


def get_top_K_components(group, K=25):
    # Sort the group by absolute 'direct_effect' in descending order
    sorted_group = group.sort_values(
        by="direct_effect", key=lambda x: x.abs(), ascending=False
    )
    # Get the top 50 components
    top_50_components = sorted_group.head(K)
    return top_50_components["name"].tolist()


def get_compo_sets_by_topK(df, K=25):
    grouped_df = df.groupby(["sample_idx", "narrative_variable"]).apply(
        partial(get_top_K_components, K=K)
    )

    # Convert the DataFrame into a list of sets
    L = [set(components) for components in grouped_df]
    return L


def filter_components(group, threshold=0.05):
    threshold = threshold * (group["ref_ld"] - group["cor_ld"]).iloc[0].numpy()
    return set(group.loc[abs(group["direct_effect"]) > threshold, "name"])


def get_compo_sets_by_effect(df, threshold=0.05):
    grouped_df = (
        df.groupby(["sample_idx", "narrative_variable"])
        .apply(partial(filter_components, threshold=threshold))
        .tolist()
    )
    return grouped_df


def get_set_list(df, K, threshold) -> list:
    if K is not None:
        return get_compo_sets_by_topK(df, K=K)
    elif threshold is not None:
        return get_compo_sets_by_effect(df, threshold=threshold)
    else:
        raise ValueError("K or threshold must be set")


def compute_average_overlap(
    df, N=1000, K: Optional[int] = 25, threshold: Optional[float] = None
):
    # Assuming your DataFrame is called 'df'
    # Group by 'sample_idx' and 'narrative_variable', and apply the function to each group

    L = get_set_list(df, K=K, threshold=threshold)
    overlaps = []
    for k in range(N):
        # Pick two random sets
        i, j = rd.sample(range(len(L)), 2)
        # Compute the overlap
        overlap = len(L[i].intersection(L[j])) / len(L[i].union(L[j]))
        # Update the average
        overlaps.append(overlap)
    return np.mean(overlaps), np.std(overlaps)


def comparative_overlap(df1, df2, K=25, threshold=None):
    L1, L2 = get_set_list(df1, K=K, threshold=threshold), get_set_list(
        df2, K=K, threshold=threshold
    )
    overlaps = []
    for k in range(len(L1)):
        overlap = len(L1[k].intersection(L2[k])) / len(L1[k].union(L2[k]))
        overlaps.append(overlap)
    return np.mean(overlaps), np.std(overlaps)


# %%
