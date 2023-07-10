# %%
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import torch
from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    Attribute,
    Entity,
    OperationDataset,
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
from causal_checker.datasets.dataset_utils import gen_dataset_family
from functools import partial
import random as rd


def gen_random_str(n=10, l=5):
    names = []
    letters = [chr(ord("A") + i) for i in range(26)] + [
        chr(ord("a") + i) for i in range(26)
    ]
    first_letters = set()
    for i in range(n):
        s = ""
        for k in range(l):
            letter = rd.sample(letters, 1)[0]
            if k == 0:
                while letter in first_letters:
                    letter = rd.sample(letters, 1)[0]
                first_letters.add(letter)
            s += letter
        names.append(s)
    return names


def gen_induction_prompt(
    name: str,
    query_strings: List[str],
    target_strings: List[str],
    tokenizer: Any,
    separator: str,
    K: int = 5,
):
    """The name stored the list of random strings to be used as queries in the prompt"""
    target_strings = rd.choices(target_strings, k=len(query_strings))

    rd.shuffle(query_strings)
    rd.shuffle(target_strings)

    pairs = list(zip(query_strings, target_strings))
    repeated_pairs = rd.choices(pairs, k=rd.randint(len(pairs) // 2, len(pairs)))
    test_pair, examples_pairs = repeated_pairs[0], repeated_pairs[1:]

    txt = ""
    for k_shot in range(K):
        for s, t in pairs:
            txt += s + separator + t + "\n"
        rd.shuffle(pairs)

    for s, t in examples_pairs:
        txt += s + separator + t + "\n"
    txt += test_pair[0] + separator

    # create the abstract representation
    entities = []
    for s, t in pairs:
        entities.append(
            Entity(
                name=t,
                attributes=[Attribute(name="follows", value=s, to_tokenize=False)],
                tokenizer=tokenizer,
                tokenize_name=True,
            )
        )  # s -> t
    query = Query(
        queried_attribute="name",
        filter_by=[Attribute(name="follows", value=test_pair[0])],
    )

    return ContextQueryPrompt(
        model_input=txt, query=query, context=entities, tokenizer=tokenizer
    )


def create_induction_dataset_same_prefix(
    nb_sample=100,
    tokenizer=None,
) -> List[OperationDataset]:
    dataset_names = ["random_dataset_" + str(i) for i in range(5)]
    kwargs_prompt_gen_fn = {}
    for name in dataset_names:
        kwargs_prompt_gen_fn[name] = {}
        kwargs_prompt_gen_fn[name]["query_strings"] = gen_random_str(n=10, l=5)
        kwargs_prompt_gen_fn[name]["target_strings"] = gen_random_str(n=40, l=5)

    return gen_dataset_family(
        partial(gen_induction_prompt, tokenizer=tokenizer, separator=":", K=5),
        dataset_prefix_name="induction_same_prefix",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
        kwargs_prompt_gen_fn=kwargs_prompt_gen_fn,
    )


# # %%
# model, tokenizer = get_model_and_tokenizer("pythia-70m")
# datasets = create_induction_dataset_same_prefix(
#     nb_sample=50,
#     tokenizer=tokenizer,
# )
# for dataset in datasets:
#     assert_dataset_perf(model, tokenizer, dataset, threshold=-1, verbose=True)
# # %%
# from causal_checker.datasets.induction_dataset import (
#     gen_random_str,
#     gen_induction_prompt,
# )


# name = ",".join(gen_random_str(n=3, l=5)) + ";" + ",".join(gen_random_str(n=30, l=5))
# prompts = [
#     gen_induction_prompt(name, tokenizer=tokenizer, separator=" -> ")
#     for i in range(100)
# ]


# dataset = OperationDataset(
#     operations=prompts,
#     name="test",
# )

# # %%
# print(datasets[0][0].model_input)
# %%
