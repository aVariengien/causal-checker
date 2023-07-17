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
import torch


def gen_dataset_family(
    prompt_gen_fn: Callable[..., ContextQueryPrompt],
    dataset_prefix_name: str,
    dataset_names: List[str],
    nb_sample: int,
    kwargs_prompt_gen_fn: Dict[str, Dict[str, Any]] = {},
    kwargs_dataset: Any = {},
) -> List[OperationDataset]:
    """Generic template to create a list of operations dataset from a prompt generation function."""
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name not in kwargs_prompt_gen_fn:
            kwargs_prompt_gen_fn[dataset_name] = {}
        operations = []
        for i in range(nb_sample):
            operations.append(
                prompt_gen_fn(dataset_name, **kwargs_prompt_gen_fn[dataset_name])
            )

        datasets.append(
            OperationDataset(operations=operations, name=dataset_name, **kwargs_dataset)
        )
    return datasets


def tokenize_dataset(
    dataset: OperationDataset,
    tokenizer: Any,
) -> torch.Tensor:
    """Tokenize a dataset with a given tokenizer."""
    dataset_str = [i.model_input for i in dataset]
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"])  # type: ignore
    return dataset_tok
