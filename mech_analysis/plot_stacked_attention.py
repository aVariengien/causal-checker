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
from causal_checker.models import get_model_and_tokenizer
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from transformer_lens import HookedTransformer
import pytest
from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
)

from causal_checker.datasets.dataset_utils import tokenize_dataset
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
from transformer_lens.hook_points import (  # Hooking utilities
    HookedRootModule,
    HookPoint,
)
from causal_checker.alignement import evaluate_model, InterchangeInterventionAccuracy
from functools import partial
import math
from swap_graphs.core import WildPosition

from transformer_lens import HookedTransformer

from pprint import pprint
import torch
import numpy as np
import plotly.express as px
import pandas as pd


from transformer_lens.utils import test_prompt
from swap_graphs.utils import printw

torch.set_grad_enabled(False)
# %%

model = HookedTransformer.from_pretrained(
    model_name="pythia-2.8b", cache_dir="/mnt/ssd-0/alex-dev/hf_models"
)



# %%


# %%

dataset = create_nanoQA_retrieval_dataset(nano_qa_dataset=nano_qa_dataset)

idx_per_story = {}

for i, x in enumerate(dataset):
    id = hash(x.model_input[:600])
    if id not in idx_per_story:
        idx_per_story[id] = []
    idx_per_story[hash(x.model_input[:600])].append(i)

# %%


cache = {}

model.cache_all(cache=cache)
model(dataset[0].model_input, prepend_bos=False)


def act_names():
    for k, a in cache.items():
        print(k, a.shape)


# %%


# %%
def get_str_toks(s: str, tokenizer: Any) -> List[str]:
    tok_ids = tokenizer(s)["input_ids"]
    l = [tokenizer.decode(i).replace(" ", "·") for i in tok_ids]
    return l


filter_toks = get_str_toks(
    "\n\nHere is a short story. Read it carefully and answer the questions below.",
    model.tokenizer,
)


def get_value_weighted_attn(layer, dataset_idx, h, end_position, cache):
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][dataset_idx][h][end_position]
    values = cache[f"blocks.{layer}.attn.hook_v"][dataset_idx, :, h]
    value_norm = torch.norm(values, dim=-1, keepdim=True).flatten()
    return (pattern * value_norm).cpu().numpy()


def plot_stacked_attention(
    cache: Dict[str, Any],
    dataset_idx: int,
    dataset: OperationDataset,
    title: str = "",
    value_weighted: bool = False,
    y_rank=3,
):
    df = []
    end_position = dataset.get_end_position().positions_from_idx([dataset_idx])[0]
    toks_sentence = dataset[dataset_idx].model_input
    pretty_toks = get_str_toks(toks_sentence, model.tokenizer)

    acc_attn_all_layers = torch.zeros(cache[f"blocks.1.attn.hook_pattern"].shape[-1])
    for i in range(model.cfg.n_layers):
        acc_attn = torch.zeros(cache[f"blocks.{i}.attn.hook_pattern"].shape[-1])
        for h in range(model.cfg.n_heads):
            if value_weighted:
                acc_attn += get_value_weighted_attn(
                    i, dataset_idx, h, end_position, cache=cache
                )
            else:
                acc_attn += cache[f"blocks.{i}.attn.hook_pattern"][dataset_idx][h][
                    end_position
                ].cpu()
        for j, tok in enumerate(pretty_toks):
            if j > 19 or j < 2:
                df.append(
                    {
                        "layer": i,
                        "token": tok + "     (" + str(j) + ")",
                        "attention": acc_attn[j].item(),
                    }
                )
        acc_attn_all_layers += acc_attn

    max_y = np.sort(acc_attn_all_layers.numpy())[-y_rank] + 5
    df = pd.DataFrame.from_records(df)
    fig = px.bar(
        df,
        x="token",
        y="attention",
        color="layer",
        title="stacked attention " + title,
        log_y=False,
        range_y=[0, max_y],
    )
    fig.show()


# %%
def steering_hook(
    z: torch.Tensor,
    hook: HookPoint,
    resid_source: torch.Tensor,
    end_position: Optional[WildPosition] = None,
):
    if end_position is None:
        assert z.shape[0] == 1
        z[0, -1, :] = resid_source
    else:
        print("yeah", z.shape)
        z[
            range(z.shape[0]),
            end_position.positions_from_idx(list(range(z.shape[0]))),
            :,
        ] = resid_source

    return z


def plot_series_stacked_attention(
    story_idx: int,
    with_hook=False,
    resid_hook_name: Optional[str] = None,
    resid_source: Optional[torch.Tensor] = None,
    max_samples=10,
):
    if with_hook:
        assert resid_hook_name is not None
        assert resid_source is not None

    idx = list(idx_per_story.keys())[story_idx]

    dataset_story = OperationDataset(
        operations=[dataset[i] for i in idx_per_story[idx]][:max_samples],
        name="unique_story",
    )
    dataset_story.get_end_position()
    story_toks = tokenize_dataset(dataset_story, model.tokenizer)
    cache = {}
    model.cache_all(cache=cache)

    if with_hook:
        model.add_hook(
            name=resid_hook_name,
            hook=partial(
                steering_hook,
                end_position=dataset_story.get_end_position(),
                resid_source=resid_source,
            ),
        )

    _ = model(story_toks)
    alreay_seen = set()
    for i in range(len(dataset_story)):
        q_var = dataset_story[i].query.filter_by[0].value
        if q_var in alreay_seen:
            continue
        alreay_seen.add(q_var)
        plot_stacked_attention(
            cache=cache,
            dataset_idx=i,
            dataset=dataset_story,
            title=dataset_story[i].query.filter_by[0].value,
            value_weighted=True,
            y_rank=1,
        )
    clean_gpu_mem()


# %% Define a hook to steer the model

source_queried_variable = "city"
source_input = ""
for idx, x in enumerate(dataset):
    if x.query.filter_by[0].value == source_queried_variable:
        source_input = x.model_input

model.reset_hooks()
cache_source = {}
model.cache_all(cache=cache_source)
model(source_input, prepend_bos=False)

layer = int(0.5 * model.cfg.n_layers)

resid_hook_name = f"blocks.{layer}.hook_resid_post"
print(f"layer: {layer}")
resid_post = cache_source[resid_hook_name]
resid_source = resid_post[0][-1]


# %%

plot_series_stacked_attention(3)
# %%
model.reset_hooks()


clean_gpu_mem()
# %%
plot_series_stacked_attention(
    3,
    with_hook=True,
    resid_hook_name=resid_hook_name,
    resid_source=resid_source,
    max_samples=5,
)


# %%


nano_qa_dataset = NanoQADataset(
    nb_samples=100,
    name="test",
    tokenizer=model.tokenizer,
    querried_variables=[
        "character_name",
        "city",
        "character_occupation",
        "season",
        "day_time",
    ],
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    evaluate_model,
    print_performance_table,
)

d = evaluate_model(model, nano_qa_dataset)
print_performance_table(d)
# %%


def zip_nano_qa_datasets(d1: NanoQADataset, d2: NanoQADataset) -> NanoQADataset:
    """If d1 is made of pairs C1, Q1 and d2 of pairs C2, Q2, then this function returns a dataset made of C2 Q1"""
    assert len(d1) == len(d2)


# %%


# %%
