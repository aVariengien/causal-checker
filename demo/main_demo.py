# %%
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset
from causal_checker.alignement import (
    CausalAlignement,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
)
from causal_checker.retrieval import Query, find_answer, CONTEXT_RETRIEVAL_CAUSAL_GRAPH
from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook
from causal_checker.models import (
    get_model_and_tokenizer,
)
from causal_checker.causal_graph import CausalGraph
from typing import List
from swap_graphs.utils import printw

# In this demo we perform interchange intervention by defining an Alignement between
# a model and a high-level causal graph.

# %%
# Import the model
model, tokenizer = get_model_and_tokenizer("gpt2-small")

# %%
# Define the dataset.
dataset = create_nanoQA_retrieval_dataset(nb_sample=10, tokenizer=tokenizer)

# A dataset is made of `ContextQueryPrompt`, objects representing single
# task instance that contain a `model_input`, the textual representation
# used as input to the LM and a `causal_graph_input`, for the high-level
# causal graph. In our case, the causal graph studied is
# `CONTEXT_RETRIEVAL_CAUSAL_GRAPH`, redifined here for clarity

# %%
# We study a question-answering task where the question prefix is uniform (i.e. doesnt
# depends on the question).
for x in dataset.operations[:2]:
    printw(x.model_input)
    print("===" * 7)

# %%
# define the high level casual graph


query = CausalGraph(name="query", output_type=Query, leaf=True)
context = CausalGraph(name="context", output_type=List, leaf=True)
nil = CausalGraph(
    name="nil", output_type=str, f=lambda context: "nil", children=[context]
)  # dummy variable that has no causal effect on the output
causal_graph = CausalGraph(
    name="output", output_type=str, f=find_answer, children=[query, context, nil]
)

# %%

# Defining the alignement.
# An alignement is a mapping from the variable in the high-level causal graph to
# component in the model. Here the end point of the mapping is defined in terms of
# hooks to access model components.

alig = CausalAlignement(
    causal_graph=causal_graph,
    hook_type="hf",  # hf for HuggingFace, the hook type to use (also support Transformer Lens)
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
# First, evaluate the model on the dataset

baseline = evaluate_model(
    dataset=dataset,
    batch_size=20,
    model=model,
    causal_graph=causal_graph,
    compute_metric=partial(
        InterchangeInterventionAccuracy, verbose=True
    ),  # verbose print the model answer, true answer for each input
    tokenizer=tokenizer,
)
print("baseline accuracy:", np.mean(baseline))
# %%

# Perform the interchange interventions by using the `check_alignement` function.

interchange_intervention_acc = check_alignement(
    alignement=alig,
    model=model,
    causal_graph=causal_graph,
    dataset=dataset,
    compute_metric=partial(
        InterchangeInterventionAccuracy,
        compute_mean=False,
        verbose=True,
        soft_matching=False,  # to match despite mismatch in space / capitalization
    ),
    variables_inter=[
        "query"
    ],  # the high-level variable at which we intervene. In pratice we always
    # intervene on one variable at a time.
    nb_inter=100,
    batch_size=10,
    verbose=True,
    tokenizer=tokenizer,
    eval_baseline=False,  # we already evaluated the baseline
)

print("Acuracy after interchange", np.mean(interchange_intervention_acc))

# %%
