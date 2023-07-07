# %%
from swap_graphs.utils import load_object
import pandas as pd
import plotly.express as px

# %%
pth = "/mnt/ssd-0/alex-dev/causal-checker/demo"
xp_name = "frosty_nash"  # "eloquent_mcnulty"
results = load_object(
    pth + "/xp_results", f"results_{xp_name}.pkl"
)  # eloquent_mcnulty, frosty_nash, youthful_wescoff (12b)

results2 = load_object(pth + "/xp_results", f"results_eloquent_mcnulty.pkl")

results3 = load_object(pth + "/xp_results", f"results_youthful_wescoff.pkl")
results = results + results2 + results3


def bound(x, vmin, vmax):
    return max(min(x, vmax), vmin)


# %%
random_guess = {
    "translation": 0.2,
    "type_hint": 0.33,
    "nanoQA_3Q": 0.2,
}

for i in range(len(results)):
    results[i]["layer_relative"] = results[i]["layer"] / results[i]["nb_layer"]
    if results[i]["baseline_acc"] <= 0.65:
        pass
    else:
        results[i]["iiac_filtered"] = results[i]["interchange_intervention_acc"]
        results[i]["relative_acc"] = (
            results[i]["interchange_intervention_acc"]
            - random_guess[results[i]["dataset"]]
        ) / (results[i]["baseline_acc"] - random_guess[results[i]["dataset"]])


# %%
df = pd.DataFrame.from_records(results)


# %%

px.line(
    df,
    x="layer_relative",
    y="relative_acc",
    color="model",
    facet_row="dataset",
    height=1600,
    title="Scaled Interchange Intervention Accuracy (0=random guess, 1=baseline acc) vs Scaled layer (0=first layer, 1=last layer)",
)
# %%
