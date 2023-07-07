# %%
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
from swap_graphs.utils import wrap_str, save_object, load_object, clean_gpu_mem
import pandas as pd
from causal_checker.hf_hooks import get_blocks

import time
from names_generator import generate_name

# %%

# pythia, _ = get_model_and_tokenizer("pythia-70m")
# gpt2, _ = get_model_and_tokenizer("gpt2-small")
# falcon, _ = get_model_and_tokenizer("falcon-7b")
# %%
dataset_gen_fn = [
    (create_translation_retrieval_dataset, "translation"),
    (create_code_type_retrieval_dataset, "type_hint"),
    (create_nanoQA_retrieval_dataset, "nanoQA_3Q"),
]

model_names = [
    # "pythia-70m",
    # "falcon-7b",
    # "falcon-7b-instruct",
    # "gpt2-small",
    # "gpt2-medium",
    # "gpt2-large",
    # "gpt2-xl",
    # "pythia-160m",
    # "pythia-410m",
    # "pythia-1b",
    # "pythia-2.8b",
    # "pythia-6.9b",
    "pythia-12b",
]
batch_size = 5


layer_increment = 1
# pd.DataFrame.from_records(results)
results = []
model = None

xp_name = generate_name()

for model_name in model_names:
    del model
    clean_gpu_mem()
    model, tokenizer = get_model_and_tokenizer(model_name)
    for dataset_fn, dataset_name in dataset_gen_fn:
        print(f"model: {model_name}, dataset: {dataset_name} (xp_name: {xp_name}))")
        nb_layer = len(get_blocks(model))
        datasets = dataset_fn(nb_sample=100, tokenizer=tokenizer)
        for dataset in datasets:
            baseline = evaluate_model(
                dataset=dataset,
                batch_size=batch_size,
                model=model,
                causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
                compute_metric=partial(
                    InterchangeInterventionAccuracy,
                    compute_mean=True,
                ),
                tokenizer=tokenizer,
            )
            t1 = time.time()
            for layer in range(1, nb_layer + 1, layer_increment):
                alig = CausalAlignement(
                    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
                    hook_type="hf",
                    model=model,
                    mapping_hf={
                        "query": residual_steam_hook_fn(
                            resid_layer=layer, position=dataset.get_end_position()
                        ),
                        "context": dummy_hook(),
                        "output": dummy_hook(),
                    },
                )

                interchange_intervention_acc = check_alignement(
                    alignement=alig,
                    model=model,
                    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
                    dataset=dataset,
                    compute_metric=partial(
                        InterchangeInterventionAccuracy,
                        compute_mean=True,
                    ),
                    variables_inter=["query"],
                    nb_inter=100,
                    batch_size=10,
                    verbose=False,
                    tokenizer=tokenizer,
                    eval_baseline=False,
                )

                results.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "layer": layer,
                        "nb_layer": nb_layer,
                        "interchange_intervention_acc": interchange_intervention_acc,
                        "baseline_acc": baseline,
                    }
                )
                save_object(results, path="./xp_results", name=f"results_{xp_name}.pkl")
            t2 = time.time()
            print(f"Time: {t2-t1} seconds")
