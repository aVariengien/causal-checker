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
import json

QUERIED_VARIABLES = {
    "cvdb": ["continent", "gender", "nationality"],
    "geography": ["capital", "language", "continent"],
}


def read_json(name: str):
    assert name in ["geography", "cvdb"]
    root_path = "/mnt/ssd-0/alex-dev/causal-checker"
    with open(root_path + f"/data/{name}.json") as f:
        d = json.loads(f.read())
    return d


def find_entity(json: List, queried_variable: str, entity_txt: str, tokenizer: Any):
    attributes = []
    prompt = None
    for x in json:
        if x["entity"] == entity_txt:
            attributes.append(
                Attribute(
                    name=x["predicate"],
                    value=" " + x["answer"],
                    tokenizer=tokenizer,
                    to_tokenize=True,
                )
            )
            if x["predicate"] == queried_variable:
                prompt = x["question"]

    assert (
        prompt is not None
    ), f"entity not found in json {entity_txt} {queried_variable}"
    entity = Entity(
        name=entity_txt,
        attributes=attributes,
        tokenizer=tokenizer,
        tokenize_name=False,  # also tokenize the non-name attributes as they'll be asked for
    )
    return prompt, entity


PROMPT_TEMPLATE = {
    "geography": """Question: What is the capital of France?
Answer: Paris

Question: {question}
Answer:""",
    "cvdb": """Question: What was the country of Freddie Mercury?
Answer: UK

Question: On which continent did Muhammad Ali live?
Answer: America

Question: {question}
Answer:""",
}


def make_factual_recall_prompt(
    tokenizer: Any, name: Literal["geography", "cvdb"], preloaded_json: List
) -> ContextQueryPrompt:
    entities = [x["entity"] for x in preloaded_json]
    entity_txt = rd.choice(entities)
    queried_variable = rd.choice(QUERIED_VARIABLES[name])

    prompt_txt, entity = find_entity(
        preloaded_json, queried_variable, entity_txt, tokenizer=tokenizer
    )

    query = Query(
        queried_attribute=queried_variable,
        filter_by=[Attribute(name="name", value=entity_txt)],
    )
    return ContextQueryPrompt(
        query=query,
        context=[entity],
        model_input=PROMPT_TEMPLATE[name].format(question=prompt_txt),
        tokenizer=tokenizer,
    )

    # TODO: maybe add few shot


def create_factual_recall_dataset(
    nb_sample=100, tokenizer=None, dataset_names: List[str] = []
) -> List[OperationDataset]:
    datasets = []
    if dataset_names == []:
        dataset_names = ["cvdb", "geography"]

    kwargs_prompt_gen_fn = {}
    for dataset_name in dataset_names:
        kwargs_prompt_gen_fn[dataset_name] = {}
        kwargs_prompt_gen_fn[dataset_name]["preloaded_json"] = read_json(dataset_name)

    return gen_dataset_family(
        partial(make_factual_recall_prompt, tokenizer),
        dataset_prefix_name="code_type_retrieval",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
        kwargs_prompt_gen_fn=kwargs_prompt_gen_fn,
        kwargs_dataset={"enforce_tokenisation": False},
    )


# %%
