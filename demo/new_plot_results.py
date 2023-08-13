# %%
from swap_graphs.utils import load_object, save_object
import pandas as pd
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from typing import Literal
import matplotlib.patches as patches
import plotly.graph_objects as go

# %%
pth = "/mnt/ssd-0/alex-dev/causal-checker/demo"
xp_name = "intelligent_nash"  # "eloquent_mcnulty" flamboyant_bassi  angry_gould
raw_results = load_object(
    pth + "/xp_results", f"results_{xp_name}.pkl"
)  # eloquent_mcnulty, frosty_nash, youthful_wescoff (12b)

# # %%

# big_xp_names = "intelligent_nash"  # "eloquent_mcnulty" flamboyant_bassi  angry_gould
# big_raw_results = load_object(
#     pth + "/xp_results", f"results_{big_xp_names}.pkl"
# )  # eloquent_mcnulty, frosty_nash, youthful_wescoff (12b)

# save_object(big_raw_results, pth + "/xp_results", f"results_intelligent_nash_save.pkl")
# # %%
# big_raw_results_filtered = []
# for r in big_raw_results:
#     if r["dataset"] not in [
#         "nanoQA_mixed_template",
#     ]:
#         big_raw_results_filtered.append(r)
# # %%
# new_results = big_raw_results_filtered + raw_results

# save_object(new_results, pth + "/xp_results", f"results_intelligent_nash.pkl")

# %%


results = []
dups = set()
for d in raw_results:
    xp = (
        d["model"],
        d["dataset_long_name"],
        d["layer"],
        d["metric_name"],
        d["hypothesis"],
    )
    if not xp in dups:
        results.append(d)
        dups.add(xp)


df = pd.DataFrame.from_records(results)


# %%
filtering = False
UseRdGuess = True


df["layer_relative"] = df["layer"] / df["nb_layer"]

if UseRdGuess:
    mask = df["metric_name"] == "token_prob"
    df.loc[mask, "random_guess"] = 0
    df = df.loc[df["baseline_mean"] >= df["random_guess"] * 1.1]

    df["normalized_metric"] = (df["results_mean"] - df["random_guess"]) / (
        df["baseline_mean"] - df["random_guess"]
    )
    df["normalized_metric_std"] = df["results_std"] / (
        df["baseline_mean"] - df["random_guess"]
    )
else:
    df["normalized_metric"] = df["results_mean"] / df["baseline_mean"]


if filtering:
    filtered_df = df.loc[(df["metric_name"] == "IIA") & (df["baseline_mean"] < 0.65)]
    pairs_to_remove = list(zip(filtered_df["model"], filtered_df["dataset"]))
    new_df = df[
        ~df.apply(lambda row: (row["model"], row["dataset"]) in pairs_to_remove, axis=1)
    ]
    df = new_df

# %% Perf table

model_names_ordered = [
    "falcon-7b",
    "falcon-7b-instruct",
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]


def plot_perf(df, metric: str, plot: bool = True):
    model_names = [
        "falcon-7b",
        "falcon-7b-instruct",
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]
    df = df[df["metric_name"] == metric]
    # Assuming your dataframe is called 'df'
    # Assuming the model names list is defined as follows:

    custom_order = pd.CategoricalDtype(categories=model_names, ordered=True)
    df["model"] = df["model"].astype(custom_order)
    df_sorted = df.sort_values(by="model")
    grouped_data = df_sorted.groupby(["model", "dataset"])

    baseline_means = grouped_data["baseline_mean"].mean().reset_index()
    pivot_table = baseline_means.pivot(
        index="dataset", columns="model", values="baseline_mean"
    )

    pivot_table = pivot_table.fillna(0)
    if plot:
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            cbar_kws={"label": "Baseline Mean"},
        )
        plt.xlabel("Model")
        plt.ylabel("Dataset")
        plt.title(f"Baseline {metric} for Models and Datasets")
        plt.show()
    return baseline_means


_ = plot_perf(df, "IIA")

# %% IIA


def get_model_dataset_pairs(
    df, dataset_type: Literal["dataset_long_name", "dataset"] = "dataset_long_name"
):
    df = df[(df["metric_name"] == "IIA")]
    df_sorted = df.sort_values(by="model")
    grouped_data = df_sorted.groupby(["model", dataset_type])
    baseline_means = grouped_data["baseline_mean"].mean().reset_index()
    filtered_pairs = baseline_means[baseline_means["baseline_mean"] > 0.7]
    model_dataset_pairs = list(
        zip(filtered_pairs["model"], filtered_pairs[dataset_type])
    )
    return model_dataset_pairs


def plot_hypothesis_metric(df, metric: str, hypothesis: str, plot: bool = True):
    model_names = [
        "falcon-7b",
        "falcon-7b-instruct",
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]
    model_dataset_pairs = get_model_dataset_pairs(df)
    df = df[(df["metric_name"] == metric) & (df["hypothesis"] == hypothesis)]

    max_results = (
        df.groupby(["dataset_long_name", "model", "dataset"])["normalized_metric"]
        .max()
        .reset_index()
    )
    filtered_max_results = max_results[
        max_results.apply(
            lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
            axis=1,
        )
    ]

    mean_max_results = (
        filtered_max_results.groupby(["dataset", "model"])["normalized_metric"]
        .mean()
        .reset_index()
    )

    # fitering

    mean_max_results["model"] = pd.Categorical(
        mean_max_results["model"], categories=model_names, ordered=True
    )

    pivot_table = mean_max_results.pivot(
        index="dataset", columns="model", values="normalized_metric"
    )

    pivot_table = pivot_table.fillna(float("nan"))
    if plot:
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            cbar_kws={"label": f"{metric} (max over layers)"},
        )
        plt.xlabel("Model")
        plt.ylabel("Dataset Name")
        plt.title(
            f"Normalized {metric} (0 = random guess, 1 = baseline perf) per dataset and Models"
        )
        plt.show()
    return mean_max_results


perf_df = plot_hypothesis_metric(df, metric="token_prob", hypothesis="Nil")

# %%


# %%

len(perf_df[perf_df["normalized_metric"] > 0.7])


# %% LAYYYER L2


def plot_layer_max_IIA(df, hypothesis: str, metric: str = "token_prob", plot=True):
    model_names = [
        "falcon-7b",
        "falcon-7b-instruct",
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]
    df = deepcopy(df)

    model_dataset_pairs = get_model_dataset_pairs(df)
    df = df[(df["metric_name"] == metric) & (df["hypothesis"] == hypothesis)]

    # max_results = (
    #     df.groupby(["dataset_long_name", "model", "layer_relative"])[
    #         "normalized_metric"
    #     ]
    #     .max()
    #     .reset_index()
    # )
    # filtered_max_results = max_results[
    #     max_results.apply(
    #         lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
    #         axis=1,
    #     )
    # ]

    # Assuming your dataframe is called 'df'

    # Step 1: Find the layer at which the maximal value for normalized_metric is reached for each pair (dataset_long_name, model)
    max_normalized_metric = df.groupby(["dataset_long_name", "model", "dataset"])[
        "normalized_metric"
    ].idxmax()

    max_values_df = df.loc[
        max_normalized_metric,
        [
            "dataset_long_name",
            "dataset",
            "model",
            "layer_relative",
            "normalized_metric",
        ],
    ]

    # Step 2: Map the dataset_long_name to the corresponding dataset and calculate the mean for each value of dataset

    # model_dataset_pairs = get_model_dataset_pairs(df)

    max_values_df = max_values_df[
        max_values_df.apply(
            lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
            axis=1,
        )
    ]
    filtered_mean_values = (
        max_values_df.groupby(["dataset", "model"])["layer_relative"]
        .mean()
        .reset_index()
    )

    filtered_mean_values["model"] = pd.Categorical(
        filtered_mean_values["model"], categories=model_names, ordered=True
    )
    df_plot = filtered_mean_values

    models = [x for x in model_names_ordered if x in list(df_plot["model"].unique())]
    df_plot["model_id"] = df_plot["model"].apply(lambda x: models.index(x))

    plot_df = []
    margin = 0.12

    for name in models:
        subset = df_plot[df_plot["model"] == name]
        already_seen = set()
        for row in subset.iterrows():
            same_layer = subset[subset["layer_relative"] == row[1]["layer_relative"]]
            if len(same_layer) > 1:
                col_idx = 0
                for col in same_layer.iterrows():
                    if not col[1]["dataset"] in already_seen:
                        already_seen.add(col[1]["dataset"])
                        plot_df.append(
                            {
                                "model": name,
                                "layer_relative": row[1]["layer_relative"],
                                "dataset": col[1]["dataset"],
                                "x": models.index(name) + col_idx * margin,
                            }
                        )
                        col_idx += 1
            else:
                plot_df.append(
                    {
                        "model": name,
                        "layer_relative": row[1]["layer_relative"],
                        "dataset": row[1]["dataset"],
                        "x": models.index(name),
                    }
                )

    plot_df = pd.DataFrame.from_records(plot_df)
    plot_df = plot_df.sort_values(by="dataset")

    if plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.scatterplot(
            x="x",
            y="layer_relative",
            hue="dataset",
            style="dataset",
            s=200,
            data=plot_df,
            ax=ax,
            alpha=0.9,
        )
        ax.set_xticks([i for i in range(len(models))])
        ax.set_xticklabels(models)
        plt.title(
            "Relative Layer (0. = first, 1. = last) with Max R2(C1) accuracy for Datasets and Models"
        )
        plt.ylim(0, 1)
        # Show the plot
        plt.show()
    return max_values_df


_ = plot_layer_max_IIA(df, hypothesis="Query", plot=True)

# %% ALL LAYERS L1 L2 L3


def get_layer_df(orig_df, layer: Literal["L1", "L2", "L3"]):
    if layer == "L2":
        return plot_layer_max_IIA(
            orig_df, hypothesis="Query", plot=False, metric="token_prob"
        )

    threshold = 0.8

    model_dataset_pairs = get_model_dataset_pairs(orig_df)
    df = orig_df[
        orig_df.apply(
            lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
            axis=1,
        )
    ]

    if layer == "L1":
        hypothesis = "Nil"
    elif layer == "L3":
        hypothesis = "Output"
    else:
        raise ValueError("layer must be L1 or L3")
    filter_df = deepcopy(df)
    filter_df = filter_df[
        (filter_df["metric_name"] == "token_prob")
        & (filter_df["hypothesis"] == hypothesis)
    ]
    filter_df = (
        filter_df.groupby(["dataset_long_name", "model", "layer"])["normalized_metric"]
        .mean()
        .reset_index()
    )
    filter_df = filter_df[filter_df["normalized_metric"] > threshold]
    filter_layers = list(
        zip(filter_df["model"], filter_df["dataset_long_name"], filter_df["layer"])
    )  # the layers where token_prob is > 0.8

    df = df[(df["metric_name"] == "IIA") & (df["hypothesis"] == hypothesis)]
    df = df[
        df.apply(
            lambda row: (row["model"], row["dataset_long_name"], row["layer"])
            in filter_layers,
            axis=1,
        )
    ]

    idx = None
    if layer == "L1":
        idx = df.groupby(["dataset_long_name", "model", "dataset"])[
            "layer_relative"
        ].idxmax()
    elif layer == "L3":
        idx = df.groupby(["dataset_long_name", "model", "dataset"])[
            "layer_relative"
        ].idxmin()

    df = df.loc[
        idx,
        [
            "dataset_long_name",
            "dataset",
            "model",
            "layer_relative",
            "normalized_metric",
        ],
    ]

    if (
        layer == "L1"
    ):  # we artifically add rows where the layer is 0 and the metric is the baseline corresponding to no interventions
        orig_df = orig_df[orig_df["metric_name"] == "IIA"]
        acc_perf = (
            orig_df.groupby(["model", "dataset_long_name", "dataset"])["baseline_mean"]
            .mean()
            .reset_index()
        )
        acc_perf = acc_perf[
            acc_perf.apply(
                lambda row: (row["model"], row["dataset_long_name"])
                in model_dataset_pairs,
                axis=1,
            )
        ]
        acc_perf = acc_perf[
            acc_perf.apply(
                lambda row: row["dataset"] in ["nanoQA_mixed_template", "nanoQA_3Q"],
                axis=1,
            )
        ]
        acc_perf["layer_relative"] = 0
        acc_perf["normalized_metric"] = acc_perf["baseline_mean"]
        acc_perf = acc_perf.drop("baseline_mean", axis=1)
        df = pd.concat([acc_perf, df], ignore_index=True)

    return df


def plot_IIA(df, layers=["L1", "L2", "L3"]):
    model_names = [
        "falcon-7b",
        "falcon-7b-instruct",
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]
    all_df = []

    for layer in layers:
        df_L = get_layer_df(df, layer=layer)
        df_L["layer"] = layer
        all_df.append(df_L)

    df = pd.concat(all_df, ignore_index=True)

    df = df.groupby(["dataset", "model"])["normalized_metric"].mean().reset_index()

    df["model"] = pd.Categorical(df["model"], categories=model_names, ordered=True)

    pivot_table = df.pivot(index="dataset", columns="model", values="normalized_metric")

    pivot_table = pivot_table.fillna(float("nan"))

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.heatmap(
        pivot_table,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        cbar_kws={"label": f"Normalized IIA"},
    )
    plt.xlabel("Model")
    plt.ylabel("Dataset Name")
    plt.title(f"Normalized IIA")
    plt.show()


plot_IIA(df, layers=["L1", "L2", "L3"])

# %%


# %%


# %%


def plot_layers_heatmap(
    df, metric: str, hypothesis: str, dataset: str, figsize=(20, 6)
):
    df = deepcopy(df)
    valid_pairs = get_model_dataset_pairs(df, dataset_type="dataset")
    models = [x for x, y in valid_pairs if y == dataset]

    df = df[
        (df["metric_name"] == metric)
        & (df["hypothesis"] == hypothesis)
        & (df["model"].isin(models))
        & (df["dataset"] == dataset)
    ]

    metric_df = (
        df.groupby(["model", "layer", "layer_relative"])["normalized_metric"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique models
    models = metric_df["model"].unique()
    ordered_models = [x for x in model_names_ordered if x in models]
    colormap = plt.cm.viridis
    # Plotting
    for i, model in enumerate(ordered_models):
        print(model)
        subset = metric_df[metric_df["model"] == model].sort_values(by="layer_relative")
        prev_height = 0
        rect_height = 0
        first_loop = True
        for _, row in subset.iterrows():
            if first_loop:
                rect_height = row["layer_relative"]
                first_loop = False
            color = colormap(row["normalized_metric"])
            rect = patches.Rectangle(
                (i, prev_height),
                1,
                row["layer_relative"],
                linewidth=0,
                edgecolor="black",
                facecolor=color,
                alpha=1.0,
            )
            ax.add_patch(rect)
            prev_height += rect_height

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm)
    cbar.set_label(f"Normalized {metric}", rotation=270, labelpad=15)

    # Set x and y axis limits and labels
    ax.set_xlim(0, len(ordered_models))
    ax.set_ylim(0, 1)
    ax.set_xticks([i + 0.5 for i in range(len(ordered_models))])
    ax.set_xticklabels(ordered_models)
    ax.set_xlabel("Model")
    ax.set_ylabel("Layer relative")
    ax.set_title(f"Normalized {metric} vs Layer relative for {hypothesis}")

    plt.show()


d = plot_layers_heatmap(
    df,
    metric="token_prob",
    hypothesis="Nil",
    dataset="nanoQA_uniform_answer_prefix",
    figsize=(10, 6),
)


# %% SCRAP PLOTS

models = [
    # "gpt2-xl",
    "pythia-2.8b",
    # "pythia-12b",
    # "falcon-7b",
    # "falcon-7b-instruct"
    # "gpt2-small",
]
metric = "IIA"  # IIA token_prob logit_diff
x_axis = "layer_relative"  # layer_relative layer
dataset_name = "nanoQA_uniform_answer_prefix"
df_filtered = df[
    (df["metric_name"] == metric)
    & (df["model"].isin(models))
    & (df["dataset"] == dataset_name)
].sort_values(
    by="layer_relative"
)  # & (df["dataset"] == "factual_recall")

# Step 1: Define the line dash styles for each unique value in 'hypothesis' column
line_dash_sequence = [
    "dash",
    "dot",
    "solid",
]

# Step 2: Plot the data using px.line with line_dash_sequence parameter
fig = px.line(
    df_filtered,
    x=x_axis,
    y="normalized_metric",  # normalized_metric
    color="model",
    line_dash="hypothesis",
    facet_row="dataset_long_name",
    height=800,
    title=f"Scaled {metric} vs Scaled layer (0=first layer, 1=last layer) on the {dataset_name} dataset",
    line_dash_sequence=line_dash_sequence,
)
fig.update_xaxes(title_text="Scaled Layer (0=first layer, 1=last layer)")

# Step 3: Show the plot
fig.show()

# %% FANCY PLOT FOR SINGLE MODEL

from itertools import product

model = "pythia-1b"
metric = "token_prob"  # IIA token_prob logit_diff
x_axis = "layer"  # layer_relative layer
y_axis = "normalized_metric"  # normalized_metric results_mean
dataset_name = "nanoQA_3Q"
df_filtered = df[
    (df["metric_name"] == metric)
    & (df["model"] == model)
    & (df["dataset"] == dataset_name)
    # & (df["hypothesis"] == "Query")
].sort_values(
    by="layer_relative"
)  # & (df["dataset"] == "factual_recall")


if y_axis == "normalized_metric":
    std_name = "normalized_metric_std"
elif y_axis == "results_mean":
    std_name = "results_std"
else:
    raise ValueError("y_axis must be normalized_metric or results_mean")

df_filtered = (
    df_filtered.groupby(["model", "dataset", "layer", "metric_name", "hypothesis"])
    .agg(
        {
            "normalized_metric": "mean",
            "normalized_metric_std": "mean",
            "results_mean": "mean",
            "results_std": "mean",
        }
    )
    .reset_index()
)


# Assuming your dataframe is called 'df_filtered'

# Step 1: Define the line dash styles for each unique value in 'hypothesis' column
line_dash_sequence = [
    "dash",
    "dot",
    "solid",
]

# Step 2: Plot the data using px.line with line_dash_sequence parameter
fig = px.line(
    df_filtered,
    x=x_axis,
    y=y_axis,  # normalized_metric
    color="hypothesis",
    # line_dash="hypothesis",
    # facet_row="dataset_long_name",
    height=600,
    title=f"Scaled {metric} vs layer on the {dataset_name} dataset for {model}",
    # line_dash_sequence=line_dash_sequence,
)

df_filtered["upper_bound"] = df_filtered[y_axis] + df_filtered[std_name] / np.sqrt(100)
df_filtered["lower_bound"] = df_filtered[y_axis] - df_filtered[std_name] / np.sqrt(100)


colors = [trace.line.color for trace in fig.data]

# Iterate over unique hypotheses and facet rows to add the uncertainty bounds
for idx, (hypothesis, dataset) in enumerate(
    product(df_filtered["hypothesis"].unique(), df_filtered["dataset"].unique())
):
    subset = df_filtered[
        (df_filtered["hypothesis"] == hypothesis) & (df_filtered["dataset"] == dataset)
    ]

    # Determine the corresponding color for this subset
    color = colors[idx]

    # Convert to RGBA with some transparency (0.2 in this case)
    color_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"

    # Lower bound with fill set to 'none'
    fig.add_traces(
        go.Scatter(
            x=subset[x_axis],
            y=subset["lower_bound"],
            showlegend=False,
            mode="lines",
            line=dict(width=0),
        )
    )

    # Upper bound with fill set to 'tonexty' to fill towards the lower bound
    fig.add_traces(
        go.Scatter(
            x=subset[x_axis],
            y=subset["upper_bound"],
            showlegend=False,
            mode="lines",
            line=dict(width=0),
            fillcolor=color_rgba,
            fill="tonexty",
        )
    )


fig.update_xaxes(title_text="Layer")

# Step 3: Show the plot
fig.show()


# %%
df_filtered = df[
    (df["metric_name"] == metric)
    & (df["hypothesis"] == "Query")
    & (
        df["model"].isin(["pythia-2.8b"])
        & (df["dataset"] == "nanoQA_uniform_answer_prefix")
    )
].sort_values(by="layer_relative")
# %%
