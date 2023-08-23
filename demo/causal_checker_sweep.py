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
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import (
    CausalAlignement,
    layer_span,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
    InterchangeInterventionLogitDiff,
    InterchangeInterventionTokenProbability,
    check_alignement_batched_graphs,
)
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
import torch
from causal_checker.retrieval import CONTEXT_RETRIEVAL_CAUSAL_GRAPH, OperationDataset

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
load_saved = False

if load_saved:
    pth = "/mnt/ssd-0/alex-dev/causal-checker/demo"
    xp_name = "romantic_hodgkin"  # "eloquent_mcnulty"
    results = load_object(
        pth + "/xp_results", f"results_{xp_name}.pkl"
    )  # eloquent_mcnulty, frosty_nash, youthful_wescoff (12b)
    saved_xps = set()
    for d in results:
        saved_xps.add((d["model"], d["dataset_long_name"], d["layer"]))
    results_df = results

else:
    saved_xps = set()
    results_df = []
    xp_name = generate_name()

dataset_gen_fn = [
    # (create_math_quantity_retrieval_dataset, "math_quantity"),
    # (create_nanoQA_uniform_answer_prefix_dataset, "nanoQA_uniform_answer_prefix"),
    # (create_nanoQA_question_first_dataset, "nanoQA_question_start"),
    # (create_factual_recall_dataset, "factual_recall"),
    #(create_induction_dataset_same_prefix, "induction_same_prefix"),
    #(create_nanoQA_mixed_template_dataset, "nanoQA_mixed_template"),
    (create_code_type_retrieval_dataset, "type_hint"),
    # (create_nanoQA_retrieval_dataset, "nanoQA_3Q"),
    #(create_translation_retrieval_dataset, "translation"),
]

model_names = [
    "/mnt/llama-2-70b-hf-2/Llama-2-70b-hf",
    # "/mnt/falcon-request-patching-2/Llama-2-7b-hf",
    # "/mnt/falcon-request-patching-2/Llama-2-13b-hf"
    # "falcon-40b",
    # "falcon-40b-instruct",
    # "gpt2-small",
    # "pythia-2.8b",
    # "pythia-1b",
    # "pythia-70m",
    # "falcon-7b",
    # "falcon-7b-instruct",
    # "gpt2-medium",
    # "gpt2-large",
    # "gpt2-xl",
    # "pythia-160m",
    # "pythia-410m",
    # "pythia-6.9b",
    # "pythia-12b",
]

dataset_idxs = None




print("========" * 7)
print(f" {len(model_names)} MODELS:")
for model_name in model_names:
    print(model_name)
print("========" * 7)
print(f" {len(dataset_gen_fn)} DATASETS:")
for dataset_fn, dataset_name in dataset_gen_fn:
    print(dataset_name)
print("=======")
if dataset_idxs is not None:
    print("WARNING: USING ONLY SUB DATASETS") 
print("=======")
if load_saved:
    print("WARNING: USING SAVED DF") 
    
metrics = {
    "IIA": partial(InterchangeInterventionAccuracy, compute_mean=False),
    "logit_diff": partial(InterchangeInterventionLogitDiff, compute_mean=False),
    "token_prob": partial(InterchangeInterventionTokenProbability, compute_mean=False),
    # "IIA_softmatch": partial(
    #     InterchangeInterventionAccuracy, compute_mean=False, soft_matching=True
    # ),
}


def graphs_alignments_variables(layer):
    alig1 = CausalAlignement(
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        hook_type="hf",
        model=model,
        mapping_hf={
            "nil": residual_steam_hook_fn(
                resid_layer=layer, position=dataset.get_end_position()
            ),
            "query": dummy_hook(),
            "context": dummy_hook(),
            "output": dummy_hook(),
        },
    )

    alig2 = CausalAlignement(
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        hook_type="hf",
        model=model,
        mapping_hf={
            "nil": dummy_hook(),
            "query": residual_steam_hook_fn(
                resid_layer=layer, position=dataset.get_end_position()
            ),
            "context": dummy_hook(),
            "output": dummy_hook(),
        },
    )

    alig3 = CausalAlignement(
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        hook_type="hf",
        model=model,
        mapping_hf={
            "nil": dummy_hook(),
            "query": dummy_hook(),
            "context": dummy_hook(),
            "output": residual_steam_hook_fn(
                resid_layer=layer, position=dataset.get_end_position()
            ),
        },
    )
    return (
        [CONTEXT_RETRIEVAL_CAUSAL_GRAPH] * 3,
        [alig1, alig2, alig3],
        [["nil"], ["query"], ["output"]],
    )


batch_size = 20

layer_increment = 1
relative_layer_increment = None

assert layer_increment is not None or relative_layer_increment is not None
# pd.DataFrame.from_records(results)
# %%
model = None
for model_idx, model_name in enumerate(model_names):
    print(
        f"Start model: {model_name} ({model_idx+1}/{len(model_names)}) (xp_name: {xp_name})"
    )
    del model
    clean_gpu_mem()
    model, tokenizer = get_model_and_tokenizer(model_name)
    for dataset_fn, dataset_name in dataset_gen_fn:
        print(" ---- ")
        print(
            f"model: {model_name}, dataset family: {dataset_name} (xp_name: {xp_name}))"
        )
        nb_layer = len(get_blocks(model))
        datasets = dataset_fn(nb_sample=100, tokenizer=tokenizer)
        if isinstance(datasets, OperationDataset):
            datasets = [datasets]

        if dataset_idxs is not None:
            print("WARNING: USING ONLY SUB DATASETS") 
            datasets = [datasets[i] for i in dataset_idxs]
            
        for dataset in datasets:
            print(f" ---- dataset name {dataset.name} ")
            baselines = {}
            for metric_name, metric in metrics.items():
                baselines[metric_name] = evaluate_model(
                    dataset=dataset,
                    batch_size=batch_size,
                    model=model,
                    causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
                    compute_metric=partial(
                        metric,
                        compute_mean=False,
                    ),
                    tokenizer=tokenizer,
                )

            t1 = time.time()

            if relative_layer_increment is not None:
                layer_increment = max(int(relative_layer_increment * nb_layer), 1)
            assert layer_increment is not None

            for layer in range(1, nb_layer + 1, layer_increment):
                if (model_name, dataset_name + "|" + dataset.name, layer) in saved_xps:
                    print(f"Skipping {model_name}, {dataset.name}, {layer}")
                    continue
                (
                    causal_graphs,
                    alignements,
                    list_variables_inter,
                ) = graphs_alignments_variables(layer)

                interchange_intervention_metrics = check_alignement_batched_graphs(
                    alignements=alignements,
                    causal_graphs=causal_graphs,
                    metrics=metrics,
                    list_variables_inter=list_variables_inter,
                    model=model,
                    dataset=dataset,
                    nb_inter=100,
                    batch_size=batch_size,
                    verbose=False,
                    tokenizer=tokenizer,
                )

                for metric_name, xp_results in interchange_intervention_metrics.items():
                    for hypothesis_results, hypothesis_name in zip(
                        xp_results, ["Nil", "Query", "Output"]
                    ):
                        results_df.append(
                            {
                                "metric_name": metric_name,
                                "model": model_name,
                                "dataset": dataset_name,
                                "dataset_long_name": dataset_name + "|" + dataset.name,
                                "layer": layer,
                                "nb_layer": nb_layer,
                                "hypothesis": hypothesis_name,
                                "all_results": hypothesis_results,
                                "results_mean": np.mean(hypothesis_results),
                                "results_std": np.std(hypothesis_results),
                                "all_baseline": baselines[metric_name],
                                "baseline_mean": np.mean(baselines[metric_name]),
                                "baseline_std": np.std(baselines[metric_name]),
                                "random_guess": dataset.compute_random_guess_accuracy()
                                if "logit_diff" not in metric_name
                                else 0.0,
                            }
                        )
                ta = time.time()
                print(f"Saving results for {model_name}, {dataset.name}, {layer}")
                save_object(
                    results_df, path="./xp_results", name=f"results_{xp_name}.pkl"
                )
                tb = time.time()
                print(f"Saved. {tb-ta:.2f} seconds")
            save_object(
                results_df, path="./xp_results", name=f"results_{xp_name}_backup.pkl"
            )
            t2 = time.time()
            print(f"Time: {t2-t1} seconds")

# %%
