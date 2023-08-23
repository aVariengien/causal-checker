# %%
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import torch
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

from causal_checker.utils import get_first_token, get_first_token_id


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
model_name = "/mnt/falcon-request-patching-2/Llama-2-7b-hf"  #
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
tokenizer.pad_token = tokenizer.eos_token
dataset = create_nanoQA_mixed_template_dataset(nb_sample=100, tokenizer=tokenizer)

# %%
model, tokenizer = get_model(model_name)
# %%

dataset = create_nanoQA_mixed_template_dataset(nb_sample=100, tokenizer=tokenizer)
dataset = dataset

x = dataset[0]

x.causal_graph_input["query"]
# %%
s = """<prefix>:</prefix>A"""
tok_s = get_first_token(tokenizer, s)
tok_id = get_first_token_id(tokenizer, s)

print(tok_s, tok_id)
# %%
get_first_token(tokenizer, """<prefix>"</prefix>architect""")
# %%


# %%
alig = CausalAlignement(
    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    hook_type="hf",  # hf for HuggingFace, the hook type to use (also support Transformer Lens)
    model=model,
    mapping_hf={
        "query": residual_steam_hook_fn(
            resid_layer=21, position=dataset.get_end_position()
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

dataset_gen_fn = [
    (create_nanoQA_mixed_template_dataset, "nanoQA_mixed_template"),
    (create_math_quantity_retrieval_dataset, "math_quantity"),
    (create_nanoQA_uniform_answer_prefix_dataset, "nanoQA_uniform_answer_prefix"),
    (create_nanoQA_question_first_dataset, "nanoQA_question_start"),
    (create_code_type_retrieval_dataset, "type_hint"),
    (create_factual_recall_dataset, "factual_recall"),
    (create_induction_dataset_same_prefix, "induction_same_prefix"),
    (create_nanoQA_retrieval_dataset, "nanoQA_3Q"),
    (create_translation_retrieval_dataset, "translation"),
]
for fn, name in dataset_gen_fn:
    dataset = fn(nb_sample=30, tokenizer=tokenizer)
    if isinstance(dataset, List):
        dataset = dataset[0]
    for metric in [
        InterchangeInterventionAccuracy,
        InterchangeInterventionLogitDiff,
        InterchangeInterventionTokenProbability,
    ]:
        baseline = evaluate_model(
            dataset=dataset,
            batch_size=10,
            model=model,
            causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
            compute_metric=partial(metric, verbose=False),
            tokenizer=tokenizer,
        )
        print(name, metric.__name__, "baseline:", np.mean(baseline))
    # baseline = evaluate_model(
    #     dataset=dataset,
    #     batch_size=10,
    #     model=model,
    #     causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    #     compute_metric=partial(
    #         InterchangeInterventionAccuracy, verbose=True
    #     ),  # verbose print the model answer, true answer for each input InterchangeInterventionTokenProbability
    #     tokenizer=tokenizer,
    # )
    # print("baseline accuracy:", np.mean(baseline))
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
