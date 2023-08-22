# %%
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import torch
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset
from causal_checker.datasets.typehint import create_code_type_retrieval_dataset
from causal_checker.datasets.quantity_retrieval import (
    create_math_quantity_retrieval_dataset,
)
from causal_checker.datasets.induction_dataset import (
    create_induction_dataset_same_prefix,
)
from causal_checker.alignement import (
    CausalAlignement,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
    InterchangeInterventionTokenProbability,
    InterchangeInterventionLogitDiff,
)
from causal_checker.retrieval import Query, find_answer, CONTEXT_RETRIEVAL_CAUSAL_GRAPH
from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook
from causal_checker.models import get_model_and_tokenizer, get_model
from causal_checker.causal_graph import CausalGraph
from typing import List
from swap_graphs.utils import printw


# %%
def get_model(
    model_name, dtype=torch.bfloat16, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# %%
model_name = "/mnt/falcon-request-patching-2/Llama-2-13b-hf"
model, tokenizer = get_model(model_name)
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_prefix_space = False
# # %%


# toks = tokenizer("<|endoftext|> Hello I am Alex!")

# s = tokenizer.decode(toks["input_ids"])
# # %%
# nano_qa_dataset = NanoQADataset(
#     nb_samples=10,
#     tokenizer=tokenizer,
#     nb_variable_values=5,
#     seed=42,
#     compute_end_pos=False,
#     querried_variables=["city", "character_name", "character_occupation"],
# )

# %%
toks = torch.tensor([tokenizer("What is my age? I am ")["input_ids"]])
output = model.generate(toks, max_new_tokens=1)
print(output)

# %%

from causal_checker.utils import get_first_token

text = "I am 40)"

# %%
get_first_token(tokenizer, "40)")
# %%
dataset = create_induction_dataset_same_prefix(nb_sample=100, tokenizer=tokenizer)
dataset = dataset[2]

# %%
alig = CausalAlignement(
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    hook_type="hf",  # hf for HuggingFace, the hook type to use (also support Transformer Lens)
    model=model,
    mapping_hf={
        "query": residual_steam_hook_fn(
            resid_layer=15, position=dataset.get_end_position()
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
    batch_size=5,
    model=model,
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
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
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    dataset=dataset,
    compute_metric=partial(
        InterchangeInterventionAccuracy,  # InterchangeInterventionTokenProbability  InterchangeInterventionLogitDiff InterchangeInterventionAccuracy
        compute_mean=False,
        verbose=True,
        # soft_matching=False,  # to match despite mismatch in space / capitalization
    ),
    variables_inter=[
        "query"
    ],  # the high-level variable at which we intervene. In pratice we always
    # intervene on one variable at a time.
    nb_inter=50,
    batch_size=5,
    verbose=True,
    tokenizer=tokenizer,
    eval_baseline=False,  # we already evaluated the baseline
)

print("Acuracy after interchange", np.mean(interchange_intervention_acc))

# %%
