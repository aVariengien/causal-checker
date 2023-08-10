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
from causal_checker.datasets.dataset_utils import gen_dataset_family

from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
import random as rd
from functools import partial
import numpy as np


MATH_PROBLEM_TEMPLATES = {
    "pencils": """<|endoftext|>Anthony has a collection of pencils. {blue_pencils} pencils are blue, {red_pencils} pencils are red, and {green_pencils} pencils are green.""",
    "recipe": """John is baking cookies. The recipe calls for {flour_cups} cups of flour, {sugar_cups} cup of sugar, and {chocolate_chips_cups} cups of chocolate chips. How many cups of ingredients in total are needed for the cookies?\n""",
    "bills": """Emily is counting her savings. She has {one_dollar_bills} one-dollar bills, {five_dollar_bills} five-dollar bills, and {ten_dollar_bills} ten-dollar bills. How much money in total does she have?\n""",
}


QUERIES = {
    "pencils": [
        (
            """ How many pencils in total are either blue or green?\nWe'll add the number of green pencils (""",
            "green_pencils",
        ),
        (
            """ How many pencils in total are either blue or red?\nWe'll add the number of red pencils (""",
            "red_pencils",
        ),
        (
            """ How many pencils in total are either blue or red?\nWe'll add the number of blue pencils (""",
            "blue_pencils",
        ),
    ],
    "recipe": [
        ("We'll add the number of cups of flour (", "flour_cups"),
        ("We'll add the number of cups of sugar (", "sugar_cups"),
        ("We'll add the number of cups of chocolate chips (", "chocolate_chips_cups"),
    ],
    "bills": [
        ("We'll multiply the number of one-dollar bills (", "one_dollar_bills"),
        ("We'll multiply the number of five-dollar bills (", "five_dollar_bills"),
        ("We'll multiply the number of ten-dollar bills (", "ten_dollar_bills"),
    ],
}


def make_quantity_retrieval_prompt(
    tokenizer: Any,
    name: str,
) -> ContextQueryPrompt:
    assert name in MATH_PROBLEM_TEMPLATES, "invalid name"

    quantities = np.random.randint(1, 10, size=3)
    while len(set(quantities)) < 3:
        quantities = np.random.randint(1, 10, size=3)

    if name == "pencils" or name == "bills":
        quantities *= 10
    quantity_names = [quantity_name for question, quantity_name in QUERIES[name]]
    quantities_dict = {
        quantity_name: str(quantity)
        for quantity_name, quantity in zip(quantity_names, quantities)
    }
    context_txt = MATH_PROBLEM_TEMPLATES[name].format(**quantities_dict)

    query_txt, queried_quantity = rd.choice(QUERIES[name])

    entities = []
    for quantity_name, quantity_value in quantities_dict.items():
        entities.append(
            Entity(
                name=quantity_name,
                attributes=[
                    Attribute(
                        value=quantity_value + ") ", name="quantity", to_tokenize=True
                    )
                ],
                tokenizer=tokenizer,
                tokenize_name=False,
            )
        )

    query = Query(
        queried_attribute="quantity",
        filter_by=[Attribute(value=queried_quantity, name="name")],
    )
    prompt = ContextQueryPrompt(
        model_input=context_txt + query_txt,
        query=query,
        context=entities,
        tokenizer=tokenizer,
    )
    return prompt


def create_math_quantity_retrieval_dataset(
    nb_sample=100, tokenizer=None, dataset_names: Optional[List[str]] = None
) -> List[OperationDataset]:
    if dataset_names is None:
        dataset_names = list(MATH_PROBLEM_TEMPLATES.keys())
    return gen_dataset_family(
        partial(make_quantity_retrieval_prompt, tokenizer),
        dataset_prefix_name="math_quantity_retrieval",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
    )
