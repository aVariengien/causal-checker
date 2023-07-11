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


def test_collision_detection():
    tokenizer = get_default_tokenizer()
    context = [
        Entity(
            name=f"{i}_entity_{i}",
            attributes=[
                Attribute(
                    value=f"very_long_attr_{i}",
                    name="decorator",
                    to_tokenize=True,  # same attribute prefix
                )
            ],
        )
        for i in range(10)
    ]
    query = Query(
        filter_by=[Attribute(name="name", value="1_entity_1")],
        queried_attribute="decorator",
    )
    prompt1 = ContextQueryPrompt(
        context=context,
        query=query,
        model_input="I am looking for entity 1",
        tokenizer=tokenizer,
    )
    with pytest.raises(ValueError):  # collision
        dataset = OperationDataset(
            operations=[prompt1],
            name="with_collision",
        )

    dataset = OperationDataset(
        operations=[prompt1],
        name="with_collision",
        check_for_collision=False,  # deactivate collision detection
    )

    context2 = [
        Entity(
            name=f"{i}_entity_{i}",
            attributes=[
                Attribute(
                    value=f"{chr(i)}very_long_attr_{i}",
                    name="decorator",
                    to_tokenize=True,  # attribute starts with a different letter
                )
            ],
        )
        for i in range(10)
    ]

    prompt2 = ContextQueryPrompt(
        context=context2,
        query=query,
        model_input="I am looking for entity 1",
        tokenizer=tokenizer,
    )

    dataset2 = OperationDataset(
        operations=[prompt2],
        name="without_collision",
    )


def test_end_of_text_detection():
    tokenizer = get_default_tokenizer()
    context = [Entity(name=f"entity_{i}") for i in range(10)]
    query = Query(filter_by=[Attribute(name="name", value="entity_1")])
    prompt1 = ContextQueryPrompt(
        context=context,
        query=query,
        model_input="I am looking for entity 1",
        tokenizer=tokenizer,
    )
    assert prompt1.model_input == "<|endoftext|>" + "I am looking for entity 1"

    prompt2 = ContextQueryPrompt(
        context=context,
        query=query,
        model_input="<|endoftext|>I am looking for entity 1",
        tokenizer=tokenizer,
    )
    assert prompt2.model_input == "<|endoftext|>" + "I am looking for entity 1"


def test_tokenize_only_name():
    tokenizer = get_default_tokenizer()
    context = [
        Entity(
            name=f"{i}_very_long_entity_name_{i}",
            tokenizer=tokenizer,
            tokenize_name=True,
        )
        for i in range(10)
    ]
    query1 = Query(
        filter_by=[Attribute(name="name", value="1_very_long_entity_name_1")]
    )
    prompt1 = ContextQueryPrompt(
        context=context,
        query=query1,
        model_input="I am looking for entity 1:",
        tokenizer=tokenizer,
    )

    with pytest.raises(ValueError):
        query2 = Query(
            queried_attribute="exists",
            filter_by=[Attribute(name="name", value="1_very_long_entity_name_1")],
        )
        prompt2 = ContextQueryPrompt(
            context=context,
            query=query2,
            model_input="I am looking for entity 1:",
            tokenizer=tokenizer,
        )


def test_tokenisation_fit():
    tokenizer = get_default_tokenizer()
    context = [
        Entity(
            name=f"{i}. very_long_entity_name_{i}",
            tokenizer=tokenizer,
            tokenize_name=True,
        )
        for i in range(10)
    ]
    query1 = Query(
        queried_attribute="name",
        filter_by=[Attribute(name="name", value="1. very_long_entity_name_1")],
    )
    with pytest.raises(ValueError):
        prompt1 = ContextQueryPrompt(
            context=context,
            query=query1,
            model_input="The name of the entity 1. is ",
            tokenizer=tokenizer,  # model input ends with a space
        )

    prompt1 = ContextQueryPrompt(
        context=context,
        query=query1,
        model_input="The name of the entity 1. is",
        tokenizer=tokenizer,  # model input doesn't end with a space
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


def assert_dataset_perf(
    model,
    tokenizer,
    dataset: OperationDataset,
    verbose=False,
    threshold: float = 0.8,
    soft_matching=False,
    batch_size=10,
):
    """a generic test for a dataset"""
    perf = evaluate_model(
        dataset=dataset,
        batch_size=batch_size,
        model=model,
        causal_graph=FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        compute_metric=partial(
            InterchangeInterventionAccuracy,
            verbose=True,
            soft_matching=soft_matching,
        ),
        tokenizer=tokenizer,
    )
    assert perf > threshold, f"Perf should be > {threshold} {perf}, {dataset.name}"
    if verbose:
        print(f"Perf {perf}, {dataset.name}")


def test_nanoQA_dataset_perf():
    model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
    dataset = create_nanoQA_retrieval_dataset(nb_sample=100, tokenizer=tokenizer)
    assert_dataset_perf(model, tokenizer, dataset)


# nano QA variations


def test_nanoQA_variations():
    model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
    fns = [
        create_nanoQA_uniform_answer_prefix_dataset,
        create_nanoQA_question_first_dataset,
        create_nanoQA_mixed_template_dataset,
        create_nanoQA_retrieval_dataset,
    ]
    for f in fns:
        assert_dataset_perf(
            model,
            tokenizer,
            f(nb_sample=50, tokenizer=tokenizer),
            verbose=False,
            threshold=0.8,
            soft_matching=True,
        )


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


# # translation dataset


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


# induction dataset


def test_induction_dataset():
    model, tokenizer = get_model_and_tokenizer("pythia-160m")
    datasets = create_induction_dataset_same_prefix(
        nb_sample=50,
        tokenizer=tokenizer,
    )
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)


# # test factual recall


def test_factual_recall_dataset():
    model, tokenizer = get_model_and_tokenizer("gpt2-xl")
    datasets = create_factual_recall_dataset(
        nb_sample=200,
        tokenizer=tokenizer,
    )
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)

    model, tokenizer = get_model_and_tokenizer("falcon-7b")
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)


# # test math quantity retreival


def test_quantity_retrieval_dataset():
    model, tokenizer = get_model_and_tokenizer("pythia-2.8b")
    datasets = create_math_quantity_retrieval_dataset(
        nb_sample=200,
        tokenizer=tokenizer,
    )
    for dataset in datasets:
        assert_dataset_perf(model, tokenizer, dataset)
