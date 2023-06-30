# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from transformer_lens import HookedTransformer

from causal_checker.retrieval_datasets import (
    create_nanoQA_retrieval_dataset,
)

from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Relation,
    Query,
    Attribute,
    Entity,
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
)


def test_attribute():
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")

    attr = Attribute(
        relation=Relation("name"), value="Thisisalongname", tokenizer=model.tokenizer
    )
    assert attr.first_token == "This"


def test_entity():
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")
    attr = Attribute(
        relation=Relation("role"), value="Thisisalongvale", tokenizer=model.tokenizer
    )
    entity = Entity(name="EntityTest", attributes=[attr], tokenizer=model.tokenizer)
    assert (
        len(entity.attributes) == 3
    ), f"Entity should have 3 attributes! {entity.attributes}"


def test_nanoQA_retrieval_dataset():
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")
    nano_qa_dataset = NanoQADataset(
        nb_samples=100,
        tokenizer=model.tokenizer,
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )
    dataset = create_nanoQA_retrieval_dataset(nano_qa_dataset)

    for prompt in dataset:
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


# %%
