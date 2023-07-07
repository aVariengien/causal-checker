# %%
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import (
    CausalAlignement,
    layer_span,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
)
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
import torch
from causal_checker.retrieval import (
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    ContextQueryPrompt,
)
from causal_checker.datasets.double_nanoQA import create_double_nanoQA_retreival_dataset
from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook, get_blocks
from causal_checker.models import get_model_and_tokenizer
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal, Union

import tqdm
import pandas as pd
import plotly.express as px

# %%
model, tokenizer = get_model_and_tokenizer("falcon-7b")

# %%

dataset = create_double_nanoQA_retreival_dataset(
    nb_sample=100,
    tokenizer=tokenizer,
    prepend_space=False,  # currently with no answer prefix
)
end_position = dataset.get_end_position()

# %%

perf = evaluate_model(
    dataset=dataset,
    batch_size=20,
    model=model,
    causal_graph=FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    compute_metric=partial(
        InterchangeInterventionAccuracy,
        position=end_position,
        verbose=True,
        soft_matching=False,
    ),
    tokenizer=tokenizer,
)
# %%
print(perf)

# %%


def locate_variable(
    variables: List[str],
    causal_graph: CausalGraph,
    model: Any,
    end_position: WildPosition,
    dataset: List[ContextQueryPrompt],
):
    all_iia = []
    for variable in variables:
        assert variable in causal_graph.get_all_nodes_names()
        print(f"Variable: {variable}")
        for layer in tqdm.tqdm(range(1, len(get_blocks(model)), 3)):
            mapping = {}
            for v in causal_graph.get_all_nodes_names():
                if v == variable:
                    mapping[v] = residual_steam_hook_fn(
                        resid_layer=layer, position=end_position
                    )
                else:
                    mapping[v] = dummy_hook()

            alig = CausalAlignement(
                causal_graph=causal_graph,
                hook_type="hf",
                model=model,
                mapping_hf=mapping,
            )

            iia = check_alignement(
                alignement=alig,
                model=model,
                causal_graph=causal_graph,
                dataset=dataset,
                compute_metric=partial(
                    InterchangeInterventionAccuracy, position=end_position
                ),
                variables_inter=[variable],
                nb_inter=50,
                batch_size=25,
                verbose=False,
                seed=42,
                tokenizer=tokenizer,
                eval_baseline=False,
            )

            all_iia.append({"iia": iia, "layer": layer, "variable": variable})
    return pd.DataFrame.from_records(all_iia)


# %%

iia_perf = locate_variable(
    variables=["query_entity_filter", "queried_relation"],
    causal_graph=FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    model=model,
    end_position=end_position,
    dataset=dataset,
)

# %%

fig = px.line(iia_perf, x="layer", y="iia", color="variable")
# %%

# add horizontal line
fig.add_hline(
    y=perf, line_dash="dot", annotation_text="baseline", annotation_position="top left"
)
