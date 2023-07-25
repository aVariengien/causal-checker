# %%
from swap_graphs.utils import load_object
import pandas as pd
import plotly.express as px

# %%
pth = "/mnt/ssd-0/alex-dev/causal-checker/demo"
xp_name = "interesting_sanderson"  # "eloquent_mcnulty"
results = load_object(
    pth + "/xp_results", f"results_{xp_name}.pkl"
)  # eloquent_mcnulty, frosty_nash, youthful_wescoff (12b)


# %%


def bound(x, vmin, vmax):
    return max(min(x, vmax), vmin)


# %%
filtering = False

df = pd.DataFrame.from_records(results)
df["layer_relative"] = df["layer"] / df["nb_layer"]
df["normalized_metric"] = (df["results_mean"] - df["random_guess"]) / (
    df["baseline_mean"] - df["random_guess"]
)

if filtering:
    filtered_df = df.loc[(df["metric_name"] == "IIA") & (df["baseline_mean"] < 0.65)]
    pairs_to_remove = list(zip(filtered_df["model"], filtered_df["dataset"]))
    new_df = df[
        ~df.apply(lambda row: (row["model"], row["dataset"]) in pairs_to_remove, axis=1)
    ]
    df = new_df


# %%

metric = "token_prob"  # IIA token_prob logit_diff

px.line(
    df[df["metric_name"] == metric],
    x="layer_relative",
    y="results_mean",
    color="model",
    line_dash="hypothesis",
    facet_row="dataset",
    height=400,
    title="Scaled Interchange Intervention Accuracy (0=random guess, 1=baseline acc) vs Scaled layer (0=first layer, 1=last layer)",
)

# %%
