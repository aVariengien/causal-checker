# %%

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
import pickle


# %%
def load_object(path, name):
    with open(os.path.join(path, name), "rb") as f:
        return pickle.load(f)


# %%

pth = "/mnt/ssd-0/alex-dev/causal-checker/demo"
xp_name = "final_data"  # "eloquent_mcnulty" flamboyant_bassi  angry_gould
# new gen: nifty_kowalevski
raw_results = load_object("./xp_results", f"results_{xp_name}.pkl")
# %%

xp_names_buggy = ["condescending_maxwell", "strange_ellis", "modest_shtern"]
xp_names = [
    "nifty_kowalevski",
    "vibrant_agnesi",
    "sleepy_sammet",
    "suspicious_banach",
    "nifty_moser",
    "gifted_cartwright",
    "romantic_hodgkin",
    "practical_brahmagupta",
    "elegant_cartwright",
    "intelligent_nash",
]

dataset_buggy = [
    "nanoQA_uniform_answer_prefix",
    "nanoQA_question_start",
    "induction_same_prefix",
    "nanoQA_mixed_template",
    "type_hint",
]

# %%

raw_results = []
for xp in xp_names:
    res = load_object("./xp_results", f"results_{xp}.pkl")
    raw_results += res
# %%
all_datasets = []
for xp in xp_names_buggy:
    res_buggy = load_object("./xp_results", f"results_{xp}.pkl")

    res_filtered = []

    for r in res_buggy:
        all_datasets.append(r["dataset_long_name"])
        if r["dataset"] not in dataset_buggy:
            res_filtered.append(r)
    raw_results += res_filtered

# %%
filtered_raw_results = []
for r in raw_results:
    if r["dataset_long_name"] not in [
        "induction_same_prefix|random_dataset_1",
        "induction_same_prefix|random_dataset_2",
    ]:
        if (
            r["dataset_long_name"] != "type_hint|banking"
            or r["model"] != "/mnt/falcon-request-patching-2/Llama-2-13b-hf"
        ):
            filtered_raw_results.append(r)


raw_results = filtered_raw_results
# %%

from swap_graphs.utils import save_object

save_object(raw_results, "./xp_results", f"results_final_data.pkl")


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
    if filtering:
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
    filtered_df = df.loc[
        (df["metric_name"] == "accuracy") & (df["baseline_mean"] < 0.65)
    ]
    pairs_to_remove = list(zip(filtered_df["model"], filtered_df["dataset"]))
    new_df = df[
        ~df.apply(lambda row: (row["model"], row["dataset"]) in pairs_to_remove, axis=1)
    ]
    df = new_df

# %%

LABEL_NAMES = {"Nil": "R2(C2)", "Query": "R1(C2)", "Output": "R1(C1)"}
df["label_name"] = df["hypothesis"].apply(lambda x: LABEL_NAMES[x])
df["metric_name"] = df["metric_name"].apply(lambda x: "accuracy" if x == "IIA" else x)


PRETTY_NAMES = {
    "nanoQA_3Q": "QA (base)",
    "nanoQA_uniform_answer_prefix": "QA (uniform prefix)",
    "nanoQA_question_start": "QA (question start)",
    "nanoQA_mixed_template": "QA (mixed template)",
    "induction_same_prefix": "Induction",
    "factual_recall": "Factual recall",
    "translation": "Translation",
    "math_quantity": "Variable binding",
    "type_hint": "Type hint",
}

PRETTY_MODEL_NAMES = {
    "/mnt/llama-2-70b-hf-2/Llama-2-70b-hf": "llama2-70b",
    "/mnt/falcon-request-patching-2/Llama-2-7b-hf": "llama2-7b",
    "/mnt/falcon-request-patching-2/Llama-2-13b-hf": "llama2-13b",
}

ordered_p_name = list(PRETTY_NAMES.values())


def prettify_dataset_names(df):
    df["dataset_pretty_name"] = df["dataset"].apply(lambda x: PRETTY_NAMES[x])
    return df


def prettify_model_names(df):
    df["model"] = df["model"].apply(
        lambda x: PRETTY_MODEL_NAMES[x] if x in PRETTY_MODEL_NAMES else x
    )
    return df


def sort_by_dataset(df):
    values = df["dataset_pretty_name"].unique()
    ordered_values = [x for x in ordered_p_name if x in values]
    custom_order_dataset = pd.CategoricalDtype(categories=ordered_values, ordered=True)
    df["dataset_pretty_name"] = df["dataset_pretty_name"].astype(custom_order_dataset)
    return df.sort_values(by="dataset_pretty_name")


model_names_ordered = [
    "falcon-7b",
    "falcon-7b-instruct",
    "falcon-40b",
    "falcon-40b-instruct",
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
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
]


def sort_by_model(df):
    values = df["model"].unique()
    ordered_values = model_names_ordered
    custom_order_dataset = pd.CategoricalDtype(categories=ordered_values, ordered=True)
    df["model"] = df["model"].astype(custom_order_dataset)
    return df.sort_values(by="model")


df = prettify_model_names(df)
df = prettify_dataset_names(df)
df = sort_by_dataset(df)
# %% Perf table


def plot_perf(df, metric: str, plot: bool = True):
    df = df[df["metric_name"] == metric]
    df_sorted = sort_by_model(df)
    grouped_data = df_sorted.groupby(["model", "dataset_pretty_name"])

    baseline_means = grouped_data["baseline_mean"].mean().reset_index()
    pivot_table = baseline_means.pivot(
        index="dataset_pretty_name", columns="model", values="baseline_mean"
    )

    # pivot_table = pivot_table.fillna(0)
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
        # plt.show()
        plt.savefig(f"figs/plot_perf.pdf", bbox_inches="tight")
    return baseline_means


_ = plot_perf(df, "logit_diff")

# %% accuracy


def get_model_dataset_pairs(
    df, dataset_type: Literal["dataset_long_name", "dataset"] = "dataset_long_name"
):
    df = df[(df["metric_name"] == "accuracy")]
    df_sorted = df.sort_values(by="model")
    grouped_data = df_sorted.groupby(["model", dataset_type])
    baseline_means = grouped_data["baseline_mean"].mean().reset_index()
    filtered_pairs = baseline_means[baseline_means["baseline_mean"] > 0.7]
    model_dataset_pairs = list(
        zip(filtered_pairs["model"], filtered_pairs[dataset_type])
    )
    return model_dataset_pairs


def plot_label_name_metric(df, metric: str, label_name: str, plot: bool = True):
    model_dataset_pairs = get_model_dataset_pairs(df)
    df = df[(df["metric_name"] == metric) & (df["label_name"] == label_name)]

    max_results = (
        df.groupby(["dataset_long_name", "model", "dataset_pretty_name"])[
            "normalized_metric"
        ]
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
        filtered_max_results.groupby(["dataset_pretty_name", "model"])[
            "normalized_metric"
        ]
        .mean()
        .reset_index()
    )

    # fitering

    mean_max_results = sort_by_model(mean_max_results)

    pivot_table = mean_max_results.pivot(
        index="dataset_pretty_name", columns="model", values="normalized_metric"
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
            f"Normalized {metric} (0 = random guess, 1 = baseline perf) for label {label_name} per dataset and models"
        )
        # plt.show()
        plt.savefig(f"figs/req_patching_{label_name}_{metric}.pdf", bbox_inches="tight")
    return mean_max_results


for label_name in ["R1(C1)", "R1(C2)", "R2(C2)"]:
    for metric in ["logit_diff", "accuracy", "token_prob"]:
        perf_df = plot_label_name_metric(df, metric=metric, label_name=label_name)

# %%

len(perf_df[perf_df["normalized_metric"] > 0.7])


# %% LAYYYER L2


def get_layer_max_metric(df, label_name: str, metric: str = "token_prob"):
    df = deepcopy(df)

    model_dataset_pairs = get_model_dataset_pairs(df)
    df = df[(df["metric_name"] == metric) & (df["label_name"] == label_name)]

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
    df["dataset_pretty_name"] = df["dataset_pretty_name"].astype(str)
    max_normalized_metric = df.groupby(
        ["dataset_long_name", "model", "dataset_pretty_name"]
    )["normalized_metric"].idxmax()

    max_values_df = df.loc[
        max_normalized_metric,
        [
            "dataset_long_name",
            "model",
            "layer_relative",
            "normalized_metric",
            "dataset_pretty_name",
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

    return max_values_df


# %% ALL LAYERS L1 L2 L3


def get_layer_df(orig_df, layer: Literal["L1", "L2", "L3"]):
    if layer == "L2":
        return get_layer_max_metric(orig_df, label_name="R1(C2)", metric="token_prob")

    threshold = 0.8

    model_dataset_pairs = get_model_dataset_pairs(orig_df)
    df = orig_df[
        orig_df.apply(
            lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
            axis=1,
        )
    ]

    if layer == "L1":
        label_name = "R2(C2)"
    elif layer == "L3":
        label_name = "R1(C1)"
    else:
        raise ValueError("layer must be L1 or L3")
    filter_df = deepcopy(df)
    filter_df = filter_df[
        (filter_df["metric_name"] == "token_prob")
        & (filter_df["label_name"] == label_name)
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

    df = df[(df["metric_name"] == "accuracy") & (df["label_name"] == label_name)]
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
        orig_df = orig_df[orig_df["metric_name"] == "accuracy"]
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
    df = prettify_dataset_names(df)
    return df


def plot_IIA(df, layers=["L1", "L2", "L3"]):
    all_df = []

    for layer in layers:
        df_L = get_layer_df(df, layer=layer)
        df_L["layer"] = layer
        all_df.append(df_L)

    df = pd.concat(all_df, ignore_index=True)

    df = (
        df.groupby(["model", "dataset_pretty_name"])["normalized_metric"]
        .mean()
        .reset_index()
    )

    df = sort_by_dataset(df)
    df = sort_by_model(df)

    pivot_table = df.pivot(
        index="dataset_pretty_name", columns="model", values="normalized_metric"
    )

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
    # plt.show()
    plt.savefig("figs/IAA.pdf", bbox_inches="tight")


plot_IIA(df, layers=["L1", "L2", "L3"])

# %%


def plot_layer_comparison(df, layer: Literal["L1", "L2", "L3"]):
    layer_df = get_layer_df(df, layer=layer)
    filtered_mean_values = (
        layer_df.groupby(["model", "dataset_pretty_name"])["layer_relative"]
        .mean()
        .reset_index()
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
                    if not col[1]["dataset_pretty_name"] in already_seen:
                        already_seen.add(col[1]["dataset_pretty_name"])
                        plot_df.append(
                            {
                                "model": name,
                                "layer_relative": row[1]["layer_relative"],
                                "dataset_pretty_name": col[1]["dataset_pretty_name"],
                                "x": models.index(name) + col_idx * margin,
                            }
                        )
                        col_idx += 1
            else:
                plot_df.append(
                    {
                        "model": name,
                        "layer_relative": row[1]["layer_relative"],
                        "dataset_pretty_name": row[1]["dataset_pretty_name"],
                        "x": models.index(name),
                    }
                )

    plot_df = pd.DataFrame.from_records(plot_df)
    plot_df = sort_by_dataset(plot_df)
    plot_df = sort_by_model(plot_df)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(
        x="x",
        y="layer_relative",
        hue="dataset_pretty_name",
        style="dataset_pretty_name",
        s=200,
        data=plot_df,
        ax=ax,
        alpha=0.9,
    )
    ax.set_xticks([i for i in range(len(models))])
    ax.set_xticklabels(models)
    plt.title(f"Relative {layer} for Datasets and Models")
    plt.ylim(-0.1, 1.1)
    # Show the plot
    # plt.show()
    plt.savefig(f"figs/layer_comparison_{layer}.pdf", bbox_inches="tight")


for layer in ["L1", "L2", "L3"]:
    plot_layer_comparison(df, layer=layer)

# %%


# %%


def plot_layers_heatmap(
    df, metric: str, label_name: str, dataset: str, figsize=(20, 6)
):
    df = deepcopy(df)
    valid_pairs = get_model_dataset_pairs(df, dataset_type="dataset")
    models = [x for x, y in valid_pairs if y == dataset]

    df = df[
        (df["metric_name"] == metric)
        & (df["label_name"] == label_name)
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
    ax.set_title(
        f"Normalized {metric} vs Layer relative for {label_name} on {PRETTY_NAMES[dataset]}"
    )

    # plt.show()
    plt.savefig(
        f"figs/layer_heatmap_{metric}_{label_name}_{dataset}.pdf", bbox_inches="tight"
    )


for label in ["R1(C1)", "R1(C2)", "R2(C2)"]:
    d = plot_layers_heatmap(
        df,
        metric="token_prob",
        label_name=label,
        dataset="nanoQA_uniform_answer_prefix",
        figsize=(10, 6),
    )


# %% SCRAP PLOTS

models = [
    # "gpt2-xl",
    # "pythia-2.8b",
    # "pythia-12b",
    "falcon-40b",
    # "falcon-7b-instruct"
    # "gpt2-small",
]
metric = "accuracy"  # accuracy token_prob logit_diff
x_axis = "layer_relative"  # layer_relative layer
dataset_name = "nanoQA_uniform_answer_prefix"
df_filtered = df[
    (df["metric_name"] == metric)
    & (df["model"].isin(models))
    & (df["dataset"] == dataset_name)
].sort_values(
    by="layer_relative"
)  # & (df["dataset"] == "factual_recall")

# Step 1: Define the line dash styles for each unique value in 'label_name' column
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
    line_dash="label_name",
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


def single_dataset_model_plot(model, metric, x_axis, y_axis, dataset_name):
    df_filtered = df[
        (df["metric_name"] == metric)
        & (df["model"] == model)
        & (df["dataset"] == dataset_name)
        # & (df["label_name"] == "R1(C2)")
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
        df_filtered.groupby(["model", "dataset", "layer", "metric_name", "label_name"])
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
    df_filtered = prettify_dataset_names(df_filtered)

    # Assuming your dataframe is called 'df_filtered'

    # Step 1: Define the line dash styles for each unique value in 'label_name' column
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
        color="label_name",
        # line_dash="label_name",
        # facet_row="dataset_long_name",
        height=600,
        title=f"Scaled {metric} vs layer on the {PRETTY_NAMES[dataset_name]} dataset for {model}",
        # line_dash_sequence=line_dash_sequence,
    )

    df_filtered["upper_bound"] = df_filtered[y_axis] + df_filtered[std_name] / np.sqrt(
        100
    )
    df_filtered["lower_bound"] = df_filtered[y_axis] - df_filtered[std_name] / np.sqrt(
        100
    )

    colors = [trace.line.color for trace in fig.data]

    # Iterate over unique hypotheses and facet rows to add the uncertainty bounds
    for idx, (label_name, dataset) in enumerate(
        product(
            df_filtered["label_name"].unique(),
            df_filtered["dataset_pretty_name"].unique(),
        )
    ):
        subset = df_filtered[
            (df_filtered["label_name"] == label_name)
            & (df_filtered["dataset_pretty_name"] == dataset)
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
    fig.update_layout(width=1000, height=600)
    fig.show()
    fig.write_image("figs/single_model_dataset_{model}_{metric}_{dataset_name}.pdf")


single_dataset_model_plot(
    model="llama2-13b",
    metric="token_prob",  # accuracy token_prob logit_diff
    x_axis="layer",  # layer_relative layer
    y_axis="normalized_metric",  # normalized_metric results_mean
    dataset_name="translation",
)

# %%
df_filtered = df[
    (df["metric_name"] == metric)
    & (df["label_name"] == "R1(C2)")
    & (
        df["model"].isin(["pythia-2.8b"])
        & (df["dataset"] == "nanoQA_uniform_answer_prefix")
    )
].sort_values(by="layer_relative")
# %%
