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


def plot_perf(df, metric: str):
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


plot_perf(df, "IIA")

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


def plot_hypothesis_metric(df, metric: str, hypothesis: str):
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


plot_hypothesis_metric(df, metric="token_prob", hypothesis="Query")


# %% LAYYYER


def plot_layer_max_IIA(df, hypothesis: str):
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
    df = df[(df["metric_name"] == "token_prob") & (df["hypothesis"] == hypothesis)]

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

    mean_values = (
        max_values_df.groupby(["dataset_long_name", "dataset", "model"])[
            "layer_relative"
        ]
        .mean()
        .reset_index()
    )

    # model_dataset_pairs = get_model_dataset_pairs(df)

    filtered_mean_values = mean_values[
        mean_values.apply(
            lambda row: (row["model"], row["dataset_long_name"]) in model_dataset_pairs,
            axis=1,
        )
    ]
    filtered_mean_values = (
        filtered_mean_values.groupby(["dataset", "model"])["layer_relative"]
        .mean()
        .reset_index()
    )

    filtered_mean_values["model"] = pd.Categorical(
        filtered_mean_values["model"], categories=model_names, ordered=True
    )

    # Step 3: Pivot the data to create the table with 'dataset' as rows and 'model' as columns
    pivot_table = filtered_mean_values.reset_index().pivot(
        index="dataset", columns="model", values="layer_relative"
    )

    # Step 4: Replace NaN with zeros (if there are datasets without any layers that reach maximal normalized_metric)

    # Step 5: Display the resulting table with color scale
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.heatmap(
        pivot_table,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        cbar_kws={"label": "Mean Normalized Layer at Max IIA"},
    )
    plt.xlabel("Model")
    plt.ylabel("Dataset")
    plt.title(
        "Relative Layer (0. = first, 1. = last) with Max IIA for Datasets and Models"
    )
    plt.show()
    return filtered_mean_values


df_plot = plot_layer_max_IIA(df, hypothesis="Query")

# %%


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

# %%
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

# %%


def plot_layers_heatmap(df, metric: str, hypothesis: str, dataset: str):
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

    fig, ax = plt.subplots(figsize=(20, 6))

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
    ax.set_title(f"Normalized {metric} vs Layer relative")

    plt.show()


d = plot_layers_heatmap(
    df, metric="token_prob", hypothesis="Query", dataset="nanoQA_3Q"
)


# %%

df[
    (df["model"] == "gpt2-small")
    & (df["metric_name"] == "IIA")
    & (df["hypothesis"] == "Query")
    & (df["dataset"] == "nanoQA_3Q")
]
# %%

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
    y="normalized_metric",  # normalized_metric
    color="hypothesis",
    # line_dash="hypothesis",
    facet_row="dataset_long_name",
    height=800,
    title=f"Scaled {metric} vs Scaled layer (0=first layer, 1=last layer) on the {dataset_name} dataset",
    # line_dash_sequence=line_dash_sequence,
)

df_filtered["upper_bound"] = df_filtered["normalized_metric"] + df_filtered["normalized_metric_std"]/np.sqrt(100)
df_filtered["lower_bound"] = df_filtered["normalized_metric"] - df_filtered["normalized_metric_std"]/np.sqrt(100)
fig.add_traces(
    go.Scatter(
        x=df_filtered[x_axis],
        y=df_filtered["upper_bound"],
        mode="lines",
        line=dict(width=0),
        name="Upper Bound",
    )
)
fig.add_trace(go.Scatter(x=df_filtered[x_axis], y=df_filtered["lower_bound"], mode='lines', line=dict(width=0), name='Lower Bound', fillcolor='rgba(0,100,80,0.2)', fill='tonexty'))


fig.update_xaxes(title_text="Scaled Layer (0=first layer, 1=last layer)")

# Step 3: Show the plot
fig.show()

# %%



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
    y="normalized_metric",  # normalized_metric
    color="hypothesis",
    # line_dash="hypothesis",
    facet_row="dataset_long_name",
    height=800,
    title=f"Scaled {metric} vs Scaled layer (0=first layer, 1=last layer) on the {dataset_name} dataset",
    # line_dash_sequence=line_dash_sequence,
)

df_filtered["upper_bound"] = df_filtered["normalized_metric"] + df_filtered["normalized_metric_std"]/np.sqrt(100)
df_filtered["lower_bound"] = df_filtered["normalized_metric"] - df_filtered["normalized_metric_std"]/np.sqrt(100)
fig.add_traces(
    go.Scatter(
        x=df_filtered[x_axis],
        y=df_filtered["upper_bound"],
        mode="lines",
        line=dict(width=0),
        name="Upper Bound",
    )
)
fig.add_trace(go.Scatter(x=df_filtered[x_axis], y=df_filtered["lower_bound"], mode='lines', line=dict(width=0), name='Lower Bound', fillcolor='rgba(0,100,80,0.2)', fill='tonexty'))


fig.update_xaxes(title_text="Scaled Layer (0=first layer, 1=last layer)")

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
