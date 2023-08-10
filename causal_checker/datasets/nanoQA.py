# %%
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


def gen_nanoQA_entities(
    nanostory: Dict[str, Any], tokenizer: Any, prepend_space: bool = True
) -> List[Entity]:
    entities = []
    for nar_var in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES:
        entity_name = " " * prepend_space + nanostory["seed"][nar_var]
        attr = Attribute(
            value=nar_var,
            name="narrative_variable",
            tokenizer=None,  # no tokenizer, values will not be recovered
            to_tokenize=False,
        )
        entities.append(
            Entity(
                name=entity_name,
                attributes=[attr],
                tokenizer=tokenizer,
                tokenize_name=True,
            )
        )
    return entities


def nanoQA_data_to_prompt(
    nano_qa_dataset: NanoQADataset,
    idx: int,
    model_input: str,
    prepend_space: bool = True,
    check_answer: bool = True,
):
    """Turn the idx th nanostory of the nanoQA dataset into a ContextQueryPrompt"""
    q_var = nano_qa_dataset.questions[idx]["querried_variable"]
    query = Query(
        queried_attribute=str("name"),
        filter_by=[Attribute(value=q_var, name=str("narrative_variable"))],
    )
    entities = gen_nanoQA_entities(
        nano_qa_dataset.nanostories[idx],
        nano_qa_dataset.tokenizer,
        prepend_space=prepend_space,
    )

    prompt = ContextQueryPrompt(
        model_input=model_input,
        query=query,
        context=entities,
        tokenizer=nano_qa_dataset.tokenizer,
    )
    if check_answer:
        assert (
            prompt.answer == nano_qa_dataset.answer_first_token_texts[idx]
        ), f"NanoQA and ContextQueryPrompt answers are different! '{prompt.answer}' vs '{nano_qa_dataset.answer_first_token_texts[idx]}'"
    return prompt


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
        prompt = nanoQA_data_to_prompt(
            nano_qa_dataset, i, model_input=nano_qa_dataset.prompts_text[i]
        )
        dataset.append(prompt)

    return OperationDataset(
        operations=dataset,
        name="nanoQA_base",
    )


# same prefix


UNIFORM_PREFIX = 'Answer: The answer is "'


UNIFORM_ANSWER_PREFIX_TEMPLATE = (
    """<|endoftext|>

Here is a short story. Read it carefully and answer the questions below with a keyword from the text. Here is the format of the answer: 'The answer is "xxx".'

{nanostory_text}

Answer the questions below.

Question: {question}

"""
    + UNIFORM_PREFIX
)


def create_nanoQA_uniform_answer_prefix_dataset(
    tokenizer: Any,
    nb_sample: int,
) -> OperationDataset:
    nano_qa_dataset = NanoQADataset(
        nb_samples=nb_sample,
        tokenizer=tokenizer,
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )
    dataset = []
    for i in range(len(nano_qa_dataset)):
        model_input = UNIFORM_ANSWER_PREFIX_TEMPLATE.format(
            nanostory_text=nano_qa_dataset.nanostories[i]["story"],
            question=nano_qa_dataset.questions[i]["question"],
        )
        prompt = nanoQA_data_to_prompt(
            nano_qa_dataset,
            i,
            model_input=model_input,
            prepend_space=False,
            check_answer=False,
        )
        dataset.append(prompt)

    return OperationDataset(
        operations=dataset,
        name="nanoQA_uniform_answer_prefix",
    )


QUESTION_FIRST_TEMPLATE = (
    """<|endoftext|>

Read the question below, then answer it after reading the story using a keyword from the text. Here is the format of the answer: 'The answer is "xxx".'

Question: {question}

Story: {nanostory_text}

"""
    + UNIFORM_PREFIX
)


def create_nanoQA_question_first_dataset(
    tokenizer: Any,
    nb_sample: int,
) -> OperationDataset:
    nano_qa_dataset = NanoQADataset(
        nb_samples=nb_sample,
        tokenizer=tokenizer,
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )
    dataset = []
    for i in range(len(nano_qa_dataset)):
        model_input = QUESTION_FIRST_TEMPLATE.format(
            nanostory_text=nano_qa_dataset.nanostories[i]["story"],
            question=nano_qa_dataset.questions[i]["question"],
        )
        prompt = nanoQA_data_to_prompt(
            nano_qa_dataset,
            i,
            model_input=model_input,
            prepend_space=False,
            check_answer=False,
        )
        dataset.append(prompt)

    return OperationDataset(
        operations=dataset,
        name="nanoQA_question_first",
    )


def create_nanoQA_mixed_template_dataset(
    tokenizer: Any,
    nb_sample: int,
) -> OperationDataset:
    """Uniform mix of the three kind of nanoQA datasets defined above"""
    operations = []

    dataset_sizes = [nb_sample // 3, nb_sample // 3, nb_sample - 2 * nb_sample // 3]

    operations += create_nanoQA_retrieval_dataset(
        tokenizer=tokenizer,
        nb_sample=dataset_sizes[0],
    ).operations

    operations += create_nanoQA_uniform_answer_prefix_dataset(
        tokenizer=tokenizer,
        nb_sample=dataset_sizes[1],
    ).operations

    operations += create_nanoQA_question_first_dataset(
        tokenizer=tokenizer,
        nb_sample=dataset_sizes[2],
    ).operations

    rd.shuffle(operations)

    return OperationDataset(
        operations=operations,
        name="nanoQA_mixed_template",
    )


# %%
