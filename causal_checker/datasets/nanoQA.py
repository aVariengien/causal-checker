from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph

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
import random as rd



def gen_nanoQA_entities(nanostory: Dict[str, Any], tokenizer: Any) -> List[Entity]:
    entities = []
    for nar_var in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES:
        entity_name = " " + nanostory["seed"][nar_var]
        attr = Attribute(
            value=nar_var,
            name=str("narrative_variable"),
            tokenizer=None,  # no tokenizer, values will not be recovered
        )
        entities.append(
            Entity(
                name=entity_name,
                attributes=[attr],
                tokenizer=tokenizer,
                only_tokenize_name=True,
            )
        )
    return entities


def create_nanoQA_retrieval_dataset(
    nano_qa_dataset: Optional[NanoQADataset] = None,
    tokenizer: Optional[Any] = None,
    nb_sample: Optional[int] = None,
) -> OperationDataset:
    """Generate a retrieval dataset from a NanoQA dataset. If the NanoQA dataset is not provided, it will be generated."""
    if nano_qa_dataset is None:
        assert tokenizer is not None
        assert nb_sample is not None
        nano_qa_dataset = NanoQADataset(
            nb_samples=nb_sample,
            tokenizer=tokenizer,
            nb_variable_values=5,
            seed=42,
            querried_variables=["city", "character_name", "character_occupation"],
        )
    else:
        assert tokenizer is None
        assert nb_sample is None
        tokenizer = nano_qa_dataset.tokenizer
        nb_sample = len(nano_qa_dataset)

    dataset = []

    for i in range(len(nano_qa_dataset)):
        q_var = nano_qa_dataset.questions[i]["querried_variable"]
        query = Query(
            queried_attribute=str("name"),
            filter_by=[Attribute(value=q_var, name=str("narrative_variable"))],
        )
        entities = gen_nanoQA_entities(nano_qa_dataset.nanostories[i], tokenizer)

        prompt = ContextQueryPrompt(
            model_input=nano_qa_dataset.prompts_text[i],
            query=query,
            context=entities,
            tokenizer=nano_qa_dataset.tokenizer,
        )

        assert (
            prompt.answer == nano_qa_dataset.answer_first_token_texts[i]
        ), f"NanoQA and ContextQueryPrompt answers are different! '{prompt.answer}' vs '{nano_qa_dataset.answer_first_token_texts[i]}'"

        dataset.append(prompt)

    return OperationDataset(
        operations=dataset,
        name="nanoQA",
    )
