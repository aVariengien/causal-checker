from typing import List, Dict, Any, Optional, Callable
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
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore


def gen_dataset_family(
    prompt_gen_fn: Callable[[str], ContextQueryPrompt],
    dataset_prefix_name: str,
    dataset_names: List[str],
    nb_sample: int,
) -> List[OperationDataset]:
    """Generic template to create a list of operations dataset from a prompt generation function."""
    datasets = []
    for dataset_name in dataset_names:
        operations = []
        for _ in range(nb_sample):
            operations.append(prompt_gen_fn(dataset_name))

        datasets.append(
            OperationDataset(
                operations=operations,
                name=f"code_type_retrieval_{dataset_name}",
            )
        )
    return datasets
