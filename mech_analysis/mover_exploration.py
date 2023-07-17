# %%

from causal_checker.datasets.typehint import create_code_type_retrieval_dataset

from causal_checker.datasets.translation import create_translation_retrieval_dataset
from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
    create_nanoQA_uniform_answer_prefix_dataset,
    create_nanoQA_question_first_dataset,
    create_nanoQA_mixed_template_dataset,
)

from causal_checker.datasets.induction_dataset import (
    create_induction_dataset_same_prefix,
)
from causal_checker.datasets.factual_recall import create_factual_recall_dataset
from causal_checker.datasets.quantity_retrieval import (
    create_math_quantity_retrieval_dataset,
)


from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal

from causal_checker.causal_graph import CausalGraph

from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
)

from causal_checker.datasets.dataset_utils import tokenize_dataset


from functools import partial
import math

import matplotlib.pyplot as plt
from pprint import pprint
import torch
import numpy as np
import plotly.express as px
import pandas as pd
import random as rd


torch.set_grad_enabled(False)


from causal_checker.models import get_model_and_tokenizer
from swap_graphs.datasets.nano_qa.nano_qa_utils import (
    check_tokenizer_nanoQA,
    print_performance_table,
)
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    pprint_nanoqa_prompt,
    evaluate_model,
)
from swap_graphs.utils import clean_gpu_mem
from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    OperationDataset,
    Attribute,
    Entity,
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
)
from swap_graphs.core import WildPosition
from swap_graphs.utils import printw
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import (  # Hooking utilities
    HookPoint,
)


def get_dataset_pair(variables: List[str], dataset_size: int, model: HookedTransformer):
    c1_q1_dataset = NanoQADataset(
        nb_samples=dataset_size,
        name=f"C1 Q1 - ({' '.join(variables)})",
        tokenizer=model.tokenizer,  # type: ignore
        querried_variables=variables,
        seed=rd.randint(0, 100000),
        nb_variable_values=5,
    )

    c2_q2_dataset = NanoQADataset(
        nb_samples=dataset_size,
        name="C2 Q2" + str(variables),
        tokenizer=model.tokenizer,  # type: ignore
        querried_variables=variables,
        seed=rd.randint(0, 100000),
        nb_variable_values=5,
    )
    return c1_q1_dataset, c2_q2_dataset


def cache_final_state_heads(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
    model: HookedTransformer,
):
    """Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer."""
    assert isinstance(hook.name, str)
    assert "hook_z" in hook.name
    z_value = z[
        range(z.shape[0]), end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    layer = int(hook.name.split(".")[1])
    out_value = torch.einsum(
        "h z d, b h z -> b h d",
        model.W_O[layer],
        z_value,
    )
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_final_state(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
):
    assert isinstance(hook.name, str)
    assert (
        ("mlp_out" in hook.name)
        or ("hook_scale" in hook.name)
        or ("resid_pre" in hook.name)
    )
    out_value = z[
        range(z.shape[0]), end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_final_attn_probs(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
):
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    out_value = z[
        range(z.shape[0]), :, end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_value_norm_hook(z: torch.Tensor, hook: HookPoint, cache: Dict):
    assert isinstance(hook.name, str)
    assert "hook_v" in hook.name
    cache[hook.name + "-norm"] = (
        torch.norm(z, dim=-1, keepdim=True).clone().detach().cpu()
    )


value_filter = lambda x: "hook_v" in x
pattern_filter = lambda x: "pattern" in x
all_filter = lambda x: True
head_output_filter = lambda x: "hook_z" in x
mlp_output_filter = lambda x: "hook_mlp_out" in x
layer_norm_scale_filter = lambda x: "ln_final.hook_scale" in x


def fill_cache(
    cache,
    model,
    nano_qa_dataset: NanoQADataset,
    mid_layer: int,
    additional_hooks: List[Tuple[str, Callable]] = [],
):
    end_position = WildPosition(position=nano_qa_dataset.word_idx["END"], label="END")
    get_final_state_head = partial(
        cache_final_state_heads,
        cache=cache,
        end_position=end_position,
        model=model,
    )
    get_final_pattern = partial(
        cache_final_attn_probs,
        cache=cache,
        end_position=end_position,
    )
    get_final_state_general = partial(
        cache_final_state,
        cache=cache,
        end_position=end_position,
    )
    model.run_with_hooks(
        nano_qa_dataset.prompts_tok,
        fwd_hooks=[
            (pattern_filter, get_final_pattern),
            (head_output_filter, get_final_state_head),
            (value_filter, partial(cache_value_norm_hook, cache=cache)),
            (mlp_output_filter, get_final_state_general),
            (layer_norm_scale_filter, get_final_state_general),
            (f"blocks.{mid_layer}.hook_resid_pre", get_final_state_general),
        ]
        + additional_hooks,
    )


# %%


def get_component_output(
    compo_type: str,
    layer: int,
    h: int,
    cache: Dict,
) -> torch.Tensor:
    if compo_type == "head":
        return cache["blocks.{}.attn.hook_z-out-END".format(layer)][:, h, :]
    elif compo_type == "mlp":
        return cache["blocks.{}.hook_mlp_out-out-END".format(layer)]
    raise ValueError("compo_type must be 'head' or 'mlp'")


def direct_logit_attribution(
    dataset: NanoQADataset,
    model: Any,
    cache: Dict,
    narrative_variable: str,
    ref_narrative_variable: Optional[str] = None,
):
    all_outputs = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads + 1, len(dataset), model.cfg.d_model
    )
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads + 1):
            compo_type = "head" if h < model.cfg.n_heads else "mlp"
            all_outputs[layer, h] = get_component_output(
                compo_type, layer, h, cache
            )  # tensor of size (dataset_size,d_model)

    if ref_narrative_variable is None:
        unembeds = torch.cat(
            [
                model.W_U[
                    :, dataset.narrative_variables_token_id[narrative_variable][i]
                ].unsqueeze(0)
                for i in range(len(dataset))
            ],
            dim=0,  # (dataset_size, d_model)
        ).cpu()
    else:
        unembeds = torch.cat(
            [
                (
                    model.W_U[
                        :, dataset.narrative_variables_token_id[narrative_variable][i]
                    ]
                    - model.W_U[
                        :,
                        dataset.narrative_variables_token_id[ref_narrative_variable][i],
                    ]
                ).unsqueeze(0)
                for i in range(len(dataset))
            ],
            dim=0,  # (dataset_size, d_model)
        ).cpu()

    layer_norms = cache["ln_final.hook_scale-out-END"].flatten()  # (dataset_size, 1)
    print(all_outputs.shape, unembeds.shape, layer_norms.shape)
    dla = torch.einsum(
        "l h b d, b d -> l h b", all_outputs, unembeds
    )  # (n_layers, n_heads+1, dataset_size)
    print(dla.shape)

    dla = torch.einsum(
        "l h b, b -> l h b", dla, 1 / layer_norms
    )  # (n_layers, n_heads+1, dataset_size)

    return dla


def get_attn_to_correct_tok(
    layer: int, h: int, dataset: NanoQADataset, cache, narrative_variable: str
):
    total_prob = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        correct_idx = dataset.narrative_variables_token_pos[narrative_variable][i]
        total_prob[i] = cache[f"blocks.{layer}.attn.hook_pattern-out-END"][i][h][
            correct_idx
        ].sum()

    return total_prob.cpu()

    # value_norm = cache[f"blocks.{layer}.attn.hook_v-norm"][dataset_idx, :, h]
    # return (pattern * value_norm).cpu().numpy()


def attention_to_nar_var_tok(
    dataset: NanoQADataset, model: Any, cache: Dict, narrative_variable: str
):
    attention = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, len(dataset)))
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            attention[layer, h] = get_attn_to_correct_tok(
                layer, h, dataset, cache, narrative_variable
            )
    return attention.numpy()


def get_str_toks(s: str, tokenizer: Any) -> List[str]:
    tok_ids = tokenizer(s)["input_ids"]
    l = [tokenizer.decode(i).replace(" ", "·") for i in tok_ids]
    return l


def steering_hook(
    z: torch.Tensor,
    hook: HookPoint,
    resid_source: torch.Tensor,
    end_position: Optional[WildPosition] = None,
):
    if end_position is None:
        assert z.shape[0] == 1
        z[0, -1, :] = resid_source.cuda()
    else:
        print("yeah", z.shape)
        z[
            range(z.shape[0]),
            end_position.positions_from_idx(list(range(z.shape[0]))),
            :,
        ] = resid_source.cuda()

    return z


def act_names(cache):
    for k, a in cache.items():
        print(k, a.shape)


def apply_steering(
    target_dataset: NanoQADataset,
    model: Any,
    cache_source: Dict,
    mid_layer: int,
):
    end_position_target = WildPosition(
        position=target_dataset.word_idx["END"], label="END"
    )
    model.add_hook(
        f"blocks.{mid_layer}.hook_resid_pre",
        partial(
            steering_hook,
            resid_source=cache_source[f"blocks.{mid_layer}.hook_resid_pre-out-END"],
            end_position=end_position_target,
        ),
    )


def get_mover_df(
    dataset: NanoQADataset,
    model: Any,
    narrative_variable: str = "correct",
    mid_layer=-1,
):
    cache = {}
    fill_cache(cache, model, dataset, mid_layer=mid_layer)
    dla = direct_logit_attribution(
        dataset, model, cache, narrative_variable=narrative_variable
    )
    attn_to_cor_tok = attention_to_nar_var_tok(
        dataset, model, cache, narrative_variable=narrative_variable
    )
    heads_dla = dla[:, : model.cfg.n_heads, :]
    df = []

    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            df.append(
                {
                    "layer": l,
                    "head": h,
                    "dla_mean": heads_dla[l, h].mean(),
                    "abs_dla_mean": heads_dla[l, h].abs().mean(),
                    "dla_std": heads_dla[l, h].std() / math.sqrt(len(dataset)),
                    "attn_mean": attn_to_cor_tok[l, h].mean(),
                    "attn_std": attn_to_cor_tok[l, h].std() / math.sqrt(len(dataset)),
                    "head_name": f"l {l} h {h}",
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
    dla_threshold = df["abs_dla_mean"].quantile(percentile)

    filtering = (df["abs_dla_mean"] > dla_threshold) & (
        df["attn_mean"] > attn_threshold
    )
    if filter_by_layer:
        filtering = filtering & (df["layer"] >= mid_layer)
    movers = df[filtering]
    return list(movers[["layer", "head"]].to_records(index=False))


def plot_movers(
    dataset: NanoQADataset,
    model: Any,
    title: str = "",
    df: Optional[pd.DataFrame] = None,
    mid_layer=-1,
):
    if df is None:
        df = get_mover_df(dataset, model, mid_layer=mid_layer)

    # plots the results with error bars

    df_plot = df[(df["abs_dla_mean"] > 0.01) & (df["attn_mean"] > 0.01)].copy()

    fig = px.scatter(
        df_plot,
        x="abs_dla_mean",
        y="attn_mean",
        error_x="dla_std",
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
model_name = "pythia-2.8b"
model = HookedTransformer.from_pretrained(
    model_name=model_name, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
)

MID_LAYER = None
if model_name == "pythia-2.8b":
    MID_LAYER = 17
else:
    raise ValueError(f"no mid layer set for {model_name}")
# %%

c1_q1_dataset, c2_q2_dataset = get_dataset_pair(
    variables=["city", "character_name", "character_occupation"],
    dataset_size=100,
    model=model,
)

c2_q1_dataset = c2_q2_dataset.question_from(c1_q1_dataset)

# %%
d = evaluate_model(model, c1_q1_dataset, batch_size=10)
print_performance_table(d)
# %%
pprint_nanoqa_prompt(c1_q1_dataset, 0, separator="|")
# %%
pprint_nanoqa_prompt(c2_q2_dataset, 0, separator="")
pprint_nanoqa_prompt(c2_q1_dataset, 0, separator="")

# %%
cache = {}
fill_cache(cache, model, c1_q1_dataset, mid_layer=MID_LAYER)


# %%
dla = direct_logit_attribution(
    c1_q1_dataset, model, cache, narrative_variable="correct"
)
dla_mean = dla.mean(dim=-1)
dla_std = dla.std(dim=-1)
px.imshow(dla_mean, aspect="auto")

# %%
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt


def plot_sorted_dla_bar_chart(
    dataset: NanoQADataset,
    model: HookedTransformer,
    cache: Dict,
    variable: str,
    alt_variable: Optional[str] = None,
    figsize=(10, 6),
    title_suffix="",
    error_bar=True,
):
    dla = direct_logit_attribution(
        dataset,
        model,
        cache,
        narrative_variable=variable,
        ref_narrative_variable=alt_variable,
    )
    dla_mean = dla.mean(dim=-1)
    dla_std = dla.std(dim=-1)
    df = []

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
                    "dla_mean": dla_mean[l, h].numpy(),
                    "abs_dla_mean": dla_mean[l, h].abs().numpy(),
                    "dla_std": dla_std[l, h].numpy(),
                }
            )
    df = pd.DataFrame(df)

    df_sorted = df.sort_values("abs_dla_mean", ascending=False)
    threshold = df_sorted["abs_dla_mean"].quantile(0.95)
    df_filtered = df_sorted[df_sorted["abs_dla_mean"] > threshold]

    x = df_filtered["name"]
    y = df_filtered["dla_mean"]
    plt.figure(figsize=figsize)
    if error_bar:
        error = df_filtered["dla_std"]
        plt.bar(x, y, yerr=error, capsize=4)
    else:
        plt.bar(df_filtered["name"], df_filtered["dla_mean"])

    plt.xlabel("Name")
    plt.ylabel("dla_mean")
    plt.title("Top 5% of Largest abs_dla_mean Entries" + title_suffix)
    plt.xticks(rotation=90)
    plt.show()
    return df


df = plot_sorted_dla_bar_chart(
    c1_q1_dataset, model, cache, variable="correct", figsize=(15, 8)
)

# %%
model.reset_hooks()

var = "character_name"
alt_var = "city"

mono_sample_dataset = NanoQADataset(
    nb_samples=100,
    tokenizer=model.tokenizer,  # type: ignore
    querried_variables=[var],
    nb_variable_values=5,
)
mono_cache = {}
fill_cache(mono_cache, model, mono_sample_dataset, mid_layer=MID_LAYER)
df = plot_sorted_dla_bar_chart(
    mono_sample_dataset,
    model,
    mono_cache,
    variable=var,
    alt_variable=alt_var,
    figsize=(15, 8),
    error_bar=False,
    title_suffix=f" - dataset: {mono_sample_dataset.name}",
)
# %%
dla = direct_logit_attribution(
    mono_sample_dataset,
    model,
    mono_cache,
    narrative_variable=var,
    ref_narrative_variable=alt_var,
)

print(dla.mean(dim=-1).sum())
# %%
pprint_nanoqa_prompt(mono_sample_dataset, 0, separator="")
# %%


def calculate_components(df, proportion=0.8):
    df_sorted = df.sort_values("abs_dla_mean", ascending=False)
    df_sorted["abs_dla_mean_pct"] = (
        df_sorted["abs_dla_mean"] / df_sorted["abs_dla_mean"].sum()
    )
    df_sorted["cumulative_pct"] = df_sorted["abs_dla_mean_pct"].cumsum()
    num_components = len(df_sorted[df_sorted["cumulative_pct"] <= proportion])
    return num_components


import pandas as pd
import matplotlib.pyplot as plt


def plot_num_components(df, proportions):
    df_sorted = df.sort_values("abs_dla_mean", ascending=False)
    df_sorted["abs_dla_mean_pct"] = (
        df_sorted["abs_dla_mean"] / df_sorted["abs_dla_mean"].sum()
    )
    df_sorted["cumulative_pct"] = df_sorted["abs_dla_mean_pct"].cumsum()
    num_components = []

    for proportion in proportions:
        components = len(df_sorted[df_sorted["cumulative_pct"] <= proportion])
        num_components.append(components)

    plt.plot(proportions, num_components, marker="o")
    plt.xlabel("Proportion of Cumulative abs_dla_mean")
    plt.ylabel("Number of Components")
    plt.title(
        "Number of Components for Different Proportions of Cumulative abs_dla_mean"
    )
    plt.show()


plot_num_components(df, np.linspace(0.0, 0.8, 50))


# %%
attn_to_cor_tok = attention_to_nar_var_tok(
    c1_q1_dataset, model, cache, narrative_variable="correct"
)

attn_mean = attn_to_cor_tok.mean(axis=-1)

px.imshow(attn_mean, aspect="auto")

# %%


plot_movers(c1_q1_dataset, model, title="Probed variable: correct", mid_layer=MID_LAYER)

# %%

# check if movers are the same no matter the question
variable_datasets = {}
variables = ["city", "character_name", "character_occupation", "season", "day_time"]
for variable in variables:
    variable_datasets[variable] = NanoQADataset(
        name=variable,
        nb_samples=100,
        nb_variable_values=5,
        querried_variables=[variable],
        tokenizer=model.tokenizer,  # type: ignore
        seed=None,
    )

# %%
movers_df = {}
for d in variable_datasets.values():
    movers_df[d.name] = get_mover_df(d, model, mid_layer=MID_LAYER)


for d in variable_datasets.values():
    fig = plot_movers(d, model, df=movers_df[d.name])
    fig.write_image(f"figs/movers_{model_name}_{d.name}.png")

# %%
movers = {}
percentile = 0.7
for d in variable_datasets.values():
    movers[d.name] = get_movers_names(
        d, model, percentile=percentile, df=movers_df[d.name]
    )

#  compute and print the matrix of intersection of movers for the three different datasets

movers_matrix = np.zeros((len(movers), len(movers)))
for i, (k1, v1) in enumerate(movers.items()):
    for j, (k2, v2) in enumerate(movers.items()):
        movers_matrix[i, j] = len(
            set([str(x) for x in v1]).intersection(set([str(x) for x in v2]))
        )

fig = px.imshow(movers_matrix, zmin=0)

# add ticks with the names of the datasets

fig.update_layout(
    xaxis=dict(
        tickvals=list(range(len(movers_matrix))),
        ticktext=variables,
    ),
    yaxis=dict(
        tickvals=list(range(len(movers_matrix))),
        ticktext=variables,
    ),
    title=f"Movers intersection matrix - percentile = {percentile}",
)
fig.show()

# %%

caches_per_qvar = {}
for d in variable_datasets.values():
    caches_per_qvar[d.name] = {}
    fill_cache(caches_per_qvar[d.name], model, d, mid_layer=MID_LAYER)


# %%
# TODO maybe a facet plot here with different color for the bars depending on the type

for d in variable_datasets.values():
    for var in ["city", "character_name", "character_occupation", "season", "day_time"]:
        plot_sorted_dla_bar_chart(
            d,
            model,
            caches_per_qvar[d.name],
            variable=var,
            figsize=(15, 8),
            title_suffix=f" - dataset: {d.name} - var: {var}",
        )
    print(" ==== ")


# %%

# try the steerer experiments city -> character_name

city_dataset = variable_datasets["city"]
char_name_dataset = variable_datasets["character_name"]


city_cache = {}
fill_cache(city_cache, model, city_dataset, mid_layer=MID_LAYER)

char_name_cache = {}
fill_cache(char_name_cache, model, char_name_dataset, mid_layer=MID_LAYER)

chimera_dataset = char_name_dataset.question_from(city_dataset)
chimera_cache = {}
fill_cache(chimera_cache, model, chimera_dataset, mid_layer=MID_LAYER)


df_target_char_name = get_mover_df(
    char_name_dataset, model, narrative_variable="character_name", mid_layer=MID_LAYER
)
df_target_city = get_mover_df(
    char_name_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)

df_source_city = get_mover_df(
    city_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)
# %%
plot_movers(
    char_name_dataset, model, df=df_target_char_name, title="Probed variable: char_name"
)
# %%
plot_movers(char_name_dataset, model, df=df_target_city, title="Probed variable: city")

# %%


# %%
model.reset_hooks()
apply_steering(
    target_dataset=char_name_dataset,
    model=model,
    cache_source=city_cache,
    mid_layer=MID_LAYER,
)

# %%

df_target_char_name_post_steering = get_mover_df(
    char_name_dataset, model, narrative_variable="character_name", mid_layer=MID_LAYER
)
df_target_city_post_steering = get_mover_df(
    char_name_dataset, model, narrative_variable="city", mid_layer=MID_LAYER
)
# %%

plot_movers(
    char_name_dataset,
    model,
    df=df_target_char_name_post_steering,
    title="Probed var.: char_name (AFTER STEERING)",
)
# %%
plot_movers(
    char_name_dataset,
    model,
    df=df_target_city_post_steering,
    title="Probed var.: city (AFTER STEERING)",
)

# %%
d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)

# %%

chimera_dataset = char_name_dataset.question_from(city_dataset)
d = evaluate_model(
    model, char_name_dataset, batch_size=100, label_nano_qa_dataset=chimera_dataset
)
print_performance_table(d)

# %%

d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)
# %%
model.reset_hooks()
d = evaluate_model(model, char_name_dataset, batch_size=100)
print_performance_table(d)

# %%
plot_movers(
    city_dataset,
    model,
    df=df_source_city,
    title="Probed variable: baseline city dataset",
)
# %%
model.reset_hooks()
d = evaluate_model(model, city_dataset, batch_size=100)
print_performance_table(d)
# %%
import plotly.graph_objects as go
import numpy as np


def plot_dataframe_comparison(df1, df2, name1="df1", name2="df2", variable_name=""):
    # Filter points based on condition
    threshold = 0.1
    filtering = ((df1["abs_dla_mean"] > threshold) & (df1["attn_mean"] > threshold)) | (
        (df2["abs_dla_mean"] > threshold) & (df2["attn_mean"] > threshold)
    )
    df1_filtered = df1[filtering]
    df2_filtered = df2[filtering]

    fig = go.Figure()
    # Plot lines connecting corresponding points
    for i in range(len(df1_filtered)):
        fig.add_trace(
            go.Scatter(
                x=[
                    df1_filtered["abs_dla_mean"].iloc[i],
                    df2_filtered["abs_dla_mean"].iloc[i],
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
            x=df1_filtered["abs_dla_mean"],
            y=df1_filtered["attn_mean"],
            mode="markers",
            name=name1,
            marker=dict(
                symbol="triangle-up",
                size=15,
                color=df1_filtered["layer"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Layer"),  # Add color bar here
            ),
            error_x=dict(
                type="data",
                array=df1_filtered["dla_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            ),
            error_y=dict(
                type="data",
                array=df1_filtered["attn_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            ),
        )
    )

    # Plot points for df2 with circles
    fig.add_trace(
        go.Scatter(
            x=df2_filtered["abs_dla_mean"],
            y=df2_filtered["attn_mean"],
            mode="markers",
            name=name2,
            marker=dict(
                symbol="circle",
                size=15,
                color=df2_filtered["layer"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Layer"),  # Add color bar here
            ),
            error_x=dict(
                type="data",
                array=df2_filtered["dla_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            ),
            error_y=dict(
                type="data",
                array=df2_filtered["attn_std"],
                visible=True,
                color="rgba(0, 0, 0, 0.3)",  # Adjust opacity here
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Absolute Direct Logit Attribution",
        yaxis_title="Attention to first token",
        title=f"Absolute Direct Logit Attribution vs Attention to correct token (variable: {variable_name})",
        height=800,
        width=1200,
        legend=dict(x=100, y=1000),  # Adjust legend position here
    )
    fig.show()


plot_dataframe_comparison(
    df_target_city,
    df_target_city_post_steering,
    name1="City (pre-patching)",
    name2="City (post-patching)",
    variable_name="city",
)
# %%
plot_dataframe_comparison(
    df_target_city,
    df_target_city_post_steering,
    name1="City (pre-patching)",
    name2="City (post-patching)",
    variable_name="city",
)
# %%

plot_dataframe_comparison(
    df_target_char_name,
    df_target_char_name_post_steering,
    name1="Char. name (pre-patching)",
    name2="Char. name (post-patching)",
    variable_name="char_name",
)

# %%
plot_dataframe_comparison(
    df_source_city,
    df_target_city_post_steering,
    name1="City (on the city-dataset)",
    name2="City (post-patching)",
    variable_name="city",
)

# %%
plot_movers(char_name_dataset, model, df=df_target_city_post_steering)
# %%

# Let's do attention patching

# %% get the movers (making the union of the city and char_name movers)

char_name_movers = get_movers_names(
    char_name_dataset,
    model,
    percentile=0.0,
    df=df_target_char_name,
    filter_by_layer=False,
    mid_layer=MID_LAYER,
)
city_movers = get_movers_names(
    city_dataset,
    model,
    percentile=0.0,
    df=df_source_city,
    filter_by_layer=False,
    mid_layer=MID_LAYER,
)
all_movers = char_name_movers
for m in city_movers:
    if not m in all_movers:
        all_movers.append(m)


print(f"Number of movers: {len(all_movers)}")
# %%


def attention_pattern_hook_precise(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    source_q_var_positions: List[List[int]],
    target_q_var_positions: List[List[int]],
):
    """
    Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer.
    Only changes the attentions pattern at the position of the first token of the target and source variable.
    """
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    layer = int(hook.name.split(".")[1])
    heads = [h for (l, h) in movers if l == layer]

    # [1, 32, 272, 272] OG pattern shape
    # hook_pattern-out-END torch.Size([100, 32, 289])
    source_pattern = source_cache[f"blocks.{layer}.attn.hook_pattern-out-END"]
    batch_size = z.shape[0]
    max_len = z.shape[2]
    end_pos = end_position.positions_from_idx(list(range(batch_size)))
    for h in heads:
        for i in range(batch_size):
            edit_pos = target_q_var_positions[i] + source_q_var_positions[i]
            if i == 0:
                print(f"target {h}")
                print(z[i, h, end_pos[i], target_q_var_positions[i]])
                print(source_pattern[i, h, target_q_var_positions[i]])
                print("source")
                print(z[i, h, end_pos[i], source_q_var_positions[i]])
                print(source_pattern[i, h, source_q_var_positions[i]])
            z[i, h, end_pos[i], edit_pos] = source_pattern[i, h, edit_pos].cuda()

    return z


def attention_pattern_hook_all_seq(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    end_of_story_pos: List[int],
):
    """Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer."""
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    layer = int(hook.name.split(".")[1])
    heads = [h for (l, h) in movers if l == layer]

    # [1, 32, 272, 272] OG pattern shape
    # hook_pattern-out-END torch.Size([100, 32, 289])
    source_pattern = source_cache[f"blocks.{layer}.attn.hook_pattern-out-END"]
    batch_size = z.shape[0]
    max_len = z.shape[2]
    for h in heads:
        for i in range(batch_size):
            z[
                range(batch_size),
                h,
                end_position.positions_from_idx(list(range(batch_size))),
                : end_of_story_pos[i],
            ] = source_pattern[:, h, : end_of_story_pos[i]].cuda()

    return z


pattern_filter = lambda x: "pattern" in x


def apply_attention_patch(
    model: HookedTransformer,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    target_dataset: NanoQADataset,
    source_dataset: NanoQADataset,
    mode: str = "precise",
):
    assert mode in ["precise", "all_seq"]
    end_position_target = WildPosition(
        position=target_dataset.word_idx["END"], label="END"
    )

    for i in range(len(target_dataset)):
        assert (
            target_dataset.nanostory_end_token_pos[i]
            == source_dataset.nanostory_end_token_pos[i]
        )

    if mode == "all_seq":
        hook_fn = partial(
            attention_pattern_hook_all_seq,
            movers=movers,
            source_cache=source_cache,
            end_position=end_position_target,
            end_of_story_pos=target_dataset.nanostory_end_token_pos,
        )
    else:
        assert source_dataset is not None
        hook_fn = partial(
            attention_pattern_hook_precise,
            movers=movers,
            source_cache=source_cache,
            end_position=end_position_target,
            source_q_var_positions=source_dataset.narrative_variables_token_pos[
                "correct"
            ],
            target_q_var_positions=target_dataset.narrative_variables_token_pos[
                "correct"
            ],
        )

    model.add_hook(pattern_filter, hook_fn)


# %%
model.reset_hooks()

apply_steering(char_name_dataset, model, city_cache, mid_layer=MID_LAYER)
# apply_attention_patch(
#     model,
#     all_movers,
#     chimera_cache,
#     char_name_dataset,
#     source_dataset=chimera_dataset,
#     mode="all_seq",
# )

# %%

logits = model(char_name_dataset.prompts_tok.cuda())
# %%
d = evaluate_model(
    model,
    char_name_dataset,
    logits=logits,
    batch_size=100,
)
print_performance_table(d)

d = evaluate_model(
    model,
    char_name_dataset,
    logits=logits,
    batch_size=100,
    label_nano_qa_dataset=chimera_dataset,
)
print_performance_table(d)
# %%

patched_cache = {}
fill_cache(patched_cache, model, char_name_dataset, mid_layer=MID_LAYER)

# %%
for var in ["character_name", "city"]:
    plot_sorted_dla_bar_chart(
        char_name_dataset,
        model,
        patched_cache,
        variable=var,
        figsize=(15, 8),
        title_suffix=f" {var} (post attn pattern patching)",
    )

# %%
model.reset_hooks()
wt_cache = {}
fill_cache(wt_cache, model, char_name_dataset, mid_layer=MID_LAYER)

# %%
for var in ["character_name", "city"]:
    plot_sorted_dla_bar_chart(
        char_name_dataset,
        model,
        wt_cache,
        variable=var,
        figsize=(15, 8),
        title_suffix=f" {var} (pre-patching)",
    )

# %%
