# %%
from causal_checker.datasets.typehint import create_code_type_retrieval_dataset
from causal_checker.datasets.translation import create_translation_retrieval_dataset
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset
from causal_checker.models import get_model_and_tokenizer
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from transformer_lens import HookedTransformer
import pytest
from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
)

from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    OperationDataset,
    Attribute,
    Entity,
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
)

from causal_checker.alignement import evaluate_model, InterchangeInterventionAccuracy
from functools import partial
import math


def get_default_tokenizer():
    model, tokenizer = get_model_and_tokenizer("pythia-70m")
    return tokenizer


def test_end_of_text_detection():
    context = [Entity(name=f"entity_{i}") for i in range(10)]
    query = Query(filter_by=[Attribute(name="name", value="entity_1")])
    prompt1 = ContextQueryPrompt(
        context=context, query=query, model_input="I am looking for entity 1"
    )
    assert prompt1.model_input == "<|endoftext|>" + "I am looking for entity 1"

    prompt2 = ContextQueryPrompt(
        context=context,
        query=query,
        model_input="<|endoftext|>I am looking for entity 1",
    )
    assert prompt2.model_input == "<|endoftext|>" + "I am looking for entity 1"


def test_tokenize_only_name():
    tokenizer = get_default_tokenizer()
    context = [
        Entity(
            name=f"{i}_very_long_entity_name_{i}",
            tokenizer=tokenizer,
            only_tokenize_name=True,
        )
        for i in range(10)
    ]
    query1 = Query(
        filter_by=[Attribute(name="name", value="1_very_long_entity_name_1")]
    )
    prompt1 = ContextQueryPrompt(
        context=context, query=query1, model_input="I am looking for entity 1"
    )

    with pytest.raises(ValueError):
        query2 = Query(
            queried_attribute="exists",
            filter_by=[Attribute(name="name", value="1_very_long_entity_name_1")],
        )
        prompt2 = ContextQueryPrompt(
            context=context, query=query2, model_input="I am looking for entity 1"
        )


def test_nanoQA_retrieval_dataset():
    tokenizer = get_default_tokenizer()
    nano_qa_dataset = NanoQADataset(
        nb_samples=100,
        tokenizer=tokenizer,  # type: ignore
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )
    dataset = create_nanoQA_retrieval_dataset(nano_qa_dataset)

    for prompt in dataset.operations:
        cg_output = CONTEXT_RETRIEVAL_CAUSAL_GRAPH.run(inputs=prompt.causal_graph_input)
        cg_output2 = FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH.run(
            inputs=prompt.causal_graph_input
        )
        print(prompt)
        assert (
            cg_output == prompt.answer
        ), f"Prompt answer is not the same as the causal graph output! {cg_output} vs {prompt.answer}"
        assert (
            cg_output2 == prompt.answer
        ), f"Prompt answer is not the same as the fined grained causal graph output! {cg_output2} vs {prompt.answer}"


def test_nanoQA_dataset_random_guess():
    tokenizer = get_default_tokenizer()
    dataset = create_nanoQA_retrieval_dataset(nb_sample=100, tokenizer=tokenizer)
    random_guess = dataset.compute_random_guess_accuracy()
    assert math.isclose(
        random_guess, 1 / 3, abs_tol=0.05
    ), f"Random guess accuracy should be 1/5 {random_guess}"


def assert_dataset_perf(model, tokenizer, dataset: OperationDataset, verbose=False):
    """a generic test for a dataset"""
    perf = evaluate_model(
        dataset=dataset,
        batch_size=10,
        model=model,
        causal_graph=FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        compute_metric=partial(
            InterchangeInterventionAccuracy,
            verbose=False,
            soft_matching=False,
        ),
        tokenizer=tokenizer,
    )
    assert perf > 0.8, f"Perf should be > 0.8 {perf}, {dataset.name}"
    if verbose:
        print(f"Perf {perf}, {dataset.name}")


def test_nanoQA_dataset_perf():
    model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
    dataset = create_nanoQA_retrieval_dataset(nb_sample=100, tokenizer=tokenizer)
    assert_dataset_perf(model, tokenizer, dataset)


# type hint


def test_typehint_dataset():
    tokenizer = get_default_tokenizer()
    datasets = create_code_type_retrieval_dataset(
        nb_sample=100,
        tokenizer=tokenizer,
        dataset_names=["geometry", "banking", "animals"],
    )

    for dataset in datasets:
        assert math.isclose(
            dataset.compute_random_guess_accuracy(), 1 / 3, abs_tol=0.05
        ), f"Random guess accuracy should be 1/5 {dataset.compute_random_guess_accuracy()}"


def test_typehint_dataset_pythia_perf():
    model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
    datasets = create_code_type_retrieval_dataset(
        nb_sample=50,
        tokenizer=tokenizer,
    )
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)


# translation dataset


def test_translation_dataset():
    tokenizer = get_default_tokenizer()
    datasets = create_translation_retrieval_dataset(
        nb_sample=100,
        tokenizer=tokenizer,
    )

    for dataset in datasets:
        assert math.isclose(
            dataset.compute_random_guess_accuracy(), 1 / 5, abs_tol=0.05
        ), f"Random guess accuracy should be 1/5 {dataset.compute_random_guess_accuracy()}"


def test_translation_dataset_falcon_perf():
    model, tokenizer = get_model_and_tokenizer("falcon-7b")
    datasets = create_translation_retrieval_dataset(
        nb_sample=50,
        tokenizer=tokenizer,
    )
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)
