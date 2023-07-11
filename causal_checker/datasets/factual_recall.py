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
from causal_checker.retrieval import detect_first_token_collision


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


def find_prompt(json: List, queried_variable: str, entity_txt: str, tokenizer: Any):
    prompt = None
    for x in json:
        if x["entity"] == entity_txt:
            if x["predicate"] == queried_variable:
                prompt = x["question"]

    assert (
        prompt is not None
    ), f"entity not found in json {entity_txt} {queried_variable}"
    return prompt


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


def to_entities(json: List[dict], tokenizer: Any) -> list[Entity]:
    attribute_dict = {}
    for x in json:
        attributes = []
        name = x["entity"]
        if not name in attribute_dict:
            attribute_dict[name] = []
        attribute_dict[name].append((x["predicate"], x["answer"]))

    entities = []
    for name, attributes in attribute_dict.items():
        entities.append(
            Entity(
                name=name,
                attributes=[
                    Attribute(
                        name=attribute_name,
                        value=" " + attribute_value,  # prepend a space
                        to_tokenize=True,
                    )
                    for (attribute_name, attribute_value) in attributes
                ],
                tokenizer=tokenizer,
                tokenize_name=False,
            )
        )
    return entities


def get_entity_from_attribute(entities: list[Entity], attribute_value: str):
    for entity in entities:
        for attribute in entity.attributes:
            if attribute.value == attribute_value:
                return entity
    raise ValueError(f"attribute {attribute_value} not found in entities")


def collision_free_entities(json: list, tokenizer: Any) -> list[Entity]:
    """A collision arrves when two entities have an attribute that shares the same first token.
    json is a list of dict with the fileds "entity", "predicate", "answer", "question".
    Return a list of entities from json that are collision free.
    """
    entities_population = set(to_entities(json, tokenizer))
    collision = detect_first_token_collision(list(entities_population))

    while collision is not None:
        val1, val2, first_tok = collision
        coliding_entity = get_entity_from_attribute(list(entities_population), val1)
        entities_population.remove(coliding_entity)
        collision = detect_first_token_collision(list(entities_population))

    return list(entities_population)


def make_factual_recall_prompt(
    tokenizer: Any,
    name: Literal["geography", "cvdb"],
    collision_free_entities: list[Entity],
    preloaded_json: List,
) -> ContextQueryPrompt:
    entity = rd.choice(collision_free_entities)
    queried_variable = rd.choice(QUERIED_VARIABLES[name])

    prompt_txt = find_prompt(
        preloaded_json, queried_variable, entity.name, tokenizer=tokenizer
    )

    query = Query(
        queried_attribute=queried_variable,
        filter_by=[Attribute(name="name", value=entity.name)],
    )
    return ContextQueryPrompt(
        query=query,
        context=[entity],
        model_input=PROMPT_TEMPLATE[name].format(question=prompt_txt),
        tokenizer=tokenizer,
    )


def create_factual_recall_dataset(
    nb_sample=100,
    tokenizer=None,
    dataset_names: List[str] = [],
    verbose=False,
) -> List[OperationDataset]:
    datasets = []
    if dataset_names == []:
        dataset_names = ["cvdb", "geography"]

    kwargs_prompt_gen_fn = {}
    for dataset_name in dataset_names:
        kwargs_prompt_gen_fn[dataset_name] = {}
        kwargs_prompt_gen_fn[dataset_name]["preloaded_json"] = read_json(dataset_name)
        kwargs_prompt_gen_fn[dataset_name][
            "collision_free_entities"
        ] = collision_free_entities(  # collision free entities are computed at run time because of the dependency on the tokenizer
            kwargs_prompt_gen_fn[dataset_name]["preloaded_json"],
            tokenizer=tokenizer,
        )
        if verbose:
            print(
                f"dataset {dataset_name} has {len(kwargs_prompt_gen_fn[dataset_name]['collision_free_entities'])} collision free entities"
            )

    return gen_dataset_family(
        partial(make_factual_recall_prompt, tokenizer),
        dataset_prefix_name="code_type_retrieval",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
        kwargs_prompt_gen_fn=kwargs_prompt_gen_fn,
    )


# %%
