# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from transformer_lens import HookedTransformer

from causal_checker.datasets.nanoQA import (
    create_nanoQA_retrieval_dataset,
)

from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Query,
    Attribute,
    Entity,
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
    FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
)


def test_attribute():
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")

    attr = Attribute(name="name", value="Thisisalongname", tokenizer=model.tokenizer)
    assert attr.first_token == "This"


def test_entity():
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")
    attr = Attribute(name="role", value="Thisisalongvale", tokenizer=model.tokenizer)
    entity = Entity(name="EntityTest", attributes=[attr], tokenizer=model.tokenizer)
    assert (
        len(entity.attributes) == 3
    ), f"Entity should have 3 attributes! {entity.attributes}"



# %%
