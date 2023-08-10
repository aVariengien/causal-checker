# %%
from causal_checker.datasets.typehint import create_code_type_retrieval_dataset
from causal_checker.datasets.translation import create_translation_retrieval_dataset
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset
from causal_checker.datasets.factual_recall import create_factual_recall_dataset
from causal_checker.datasets.nanoQA import create_nanoQA_mixed_template_dataset
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import (
    CausalAlignement,
    layer_span,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
    InterchangeInterventionTokenProbability,
    InterchangeInterventionLogitDiff,
)
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
import torch
from causal_checker.retrieval import CONTEXT_RETRIEVAL_CAUSAL_GRAPH, find_answer

from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook
from causal_checker.models import (
    get_falcon_model,
    get_gpt2_model,
    get_pythia_model,
    get_model_and_tokenizer,
)
from swap_graphs.utils import wrap_str, save_object, load_object, printw
import pandas as pd
from causal_checker.hf_hooks import get_blocks

# %%
model, tokenizer = get_model_and_tokenizer("pythia-6.9b")


# %%
dataset = create_nanoQA_mixed_template_dataset(nb_sample=100, tokenizer=tokenizer)
if isinstance(dataset, list):
    dataset = dataset[0]

# %%

for x in dataset.operations[:10]:
    printw(x.model_input)
    print("===" * 7)


# %%


alig = CausalAlignement(
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    hook_type="hf",
    model=model,
    mapping_hf={
        "query": residual_steam_hook_fn(
            resid_layer=16, position=dataset.get_end_position()
        ),
        "context": dummy_hook(),
        "output": residual_steam_hook_fn(
            resid_layer=1, position=dataset.get_end_position()
        ),
        "nil": residual_steam_hook_fn(
            resid_layer=1, position=dataset.get_end_position()
        ),
    },
)

# %%

baseline = evaluate_model(
    dataset=dataset,
    batch_size=20,
    model=model,
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    compute_metric=partial(InterchangeInterventionAccuracy, verbose=True),
    tokenizer=tokenizer,
)
print(np.mean(baseline))
# %%


baseline, interchange_intervention_acc = check_alignement(
    alignement=alig,
    model=model,
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    dataset=dataset,
    compute_metric=partial(
        InterchangeInterventionAccuracy,
        compute_mean=False,
        verbose=True,
        soft_matching=False,
    ),
    variables_inter=["query"],
    nb_inter=100,
    batch_size=10,
    verbose=True,
    tokenizer=tokenizer,
    eval_baseline=True,
)

# baseline_percentage, iia = np.count_nonzero(baseline), np.count_nonzero(
#     interchange_intervention_acc
# )

print(np.mean(baseline), np.mean(interchange_intervention_acc))
# %%
x3 = residual_steam_hook_fn(resid_layer=2, position=dataset.get_end_position())
# %%
x2 = residual_steam_hook_fn(resid_layer=2, position=dataset.get_end_position())

# %%
x2.__code__.co_code == x3.__code__.co_code
# %%
x2.__code__.co_consts == x3.__code__.co_consts
# %%
x2.__closure__ == x3.__closure__
# %%
fn_eq(x2, x3)
# %%
