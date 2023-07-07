# %%
from causal_checker.datasets.dataset_utils import get_end_position
from causal_checker.datasets.typehint import create_code_type_retrieval_dataset
from causal_checker.datasets.translation import create_translation_retrieval_dataset
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset

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
)

from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook
from causal_checker.models import (
    get_falcon_model,
    get_gpt2_model,
    get_pythia_model,
    get_model_and_tokenizer,
)
from swap_graphs.utils import wrap_str, save_object, load_object
import pandas as pd
from causal_checker.hf_hooks import get_blocks

# %%
model, tokenizer = get_pythia_model("2.8b", dtype=torch.bfloat16)
# model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
# %%
dataset = create_code_type_retrieval_dataset(nb_sample=100, tokenizer=tokenizer)
end_position = get_end_position(dataset, tokenizer)
# %%

alig = CausalAlignement(
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    hook_type="hf",
    model=model,
    mapping_hf={
        "query": residual_steam_hook_fn(resid_layer=16, position=end_position),
        "context": dummy_hook(),
        "output": dummy_hook(),
    },
)

# %%

baseline = evaluate_model(
    dataset=dataset,
    batch_size=20,
    model=model,
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    compute_metric=partial(InterchangeInterventionAccuracy, position=end_position),
    tokenizer=tokenizer,
)

# %%

print(baseline)

# %%

idx = {f"NAME{i}": [] for i in range(1, 6)}
# %%
for i, d in enumerate(dataset):
    idx[d.query.filter_by[0].value].append(i)


# %%
baseline = torch.tensor(baseline, dtype=torch.float32)
for i in range(1, 6):
    print(baseline[idx[f"NAME{i}"]].mean())

# %%
baseline, interchange_intervention_acc = check_alignement(
    alignement=alig,
    model=model,
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    dataset=dataset,
    compute_metric=partial(InterchangeInterventionAccuracy, position=end_position),
    variables_inter=["query"],
    nb_inter=100,
    batch_size=10,
    verbose=True,
    tokenizer=tokenizer,
)

baseline_percentage, iia = np.count_nonzero(baseline), np.count_nonzero(
    interchange_intervention_acc
)
# %%
print(baseline_percentage, iia)
# %%
