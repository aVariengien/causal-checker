# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset

import torch
from causal_checker.utils import get_first_token


@define
class CausalInput:
    causal_graph_input: Dict[str, Any] = field()
    model_input: str = field()


@define
class Relation:
    relation: str = field()


@define
class Attribute:
    value: str = field()
    first_token: str = field(init=False)
    relation: Relation = field()
    tokenizer: Any = field(init=True, default=None)

    def __attrs_post_init__(self):
        self.first_token = get_first_token(self.tokenizer, self.value)

    def __hash__(self):
        return hash(self.value + "---" + self.relation.relation)

    def __eq__(self, other):
        assert isinstance(other, Attribute)
        return self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.relation.relation}={self.value} (first_tok={self.first_token}))"

    def __repr__(self) -> str:
        return self.__str__()


@define
class Entity:
    name: str = field()
    attributes: List[Attribute] = field(default=[])
    tokenizer: Any = field(init=True, default=None)
    only_tokenize_name: bool = field(default=False)

    def __attrs_post_init__(self):
        self.attributes.append(
            Attribute(
                relation=Relation("name"), value=self.name, tokenizer=self.tokenizer
            )
        )
        self.attributes.append(Attribute(relation=Relation("exists"), value="yes"))
        for a in self.attributes:
            assert isinstance(a, Attribute)
            assert isinstance(a.relation, Relation)
            assert isinstance(a.value, str)
            if not self.only_tokenize_name:
                if a.tokenizer is None:
                    a.tokenizer = self.tokenizer
                    a.first_token = get_first_token(self.tokenizer, a.value)
                assert a.first_token == get_first_token(
                    self.tokenizer, a.value
                ), "Incompatible tokenizer for entity and attribute"


@define
class Query:
    """A query is asking for the value of an attribute defined by the relation `query`. The entities find are the one that have all the attributes `filter_by`."""

    queried_relation: Relation = field(default=Relation("name"))
    filter_by: List[Attribute] = field(
        default=[Attribute(relation=Relation("exists"), value="yes")]
    )


def find_answer(query: Query, context: List[Entity]):
    """Apply the query on the context. Retruns the first token of the answer. Returns an error if no answer is found or if multiple answers are found."""
    answer_found = False
    answer = "NotFound!"
    for entity in context:
        if set(query.filter_by).issubset(
            set(entity.attributes)
        ):  # all attributes in filter_by are in entity.attributes
            for attribute in entity.attributes:
                if attribute.relation == query.queried_relation:
                    if answer_found:
                        raise ValueError(
                            f"Multiple answers found ({answer} and {entity.attributes[0].first_token})"
                        )
                    answer = attribute.first_token
                    answer_found = True
    if not answer_found:
        raise ValueError(f"No answer was found! {query} {context}")
    return answer


def find_entity(query: Query, context: List[Entity]):
    """Returns the enitity that matches the filter_by attribute of the query. Returns an error if no answer is found or if multiple answers are found."""
    entity_found = False
    queried_entity = None
    for entity in context:
        if set(query.filter_by).issubset(
            set(entity.attributes)
        ):  # all attributes in filter_by are in entity.attributes
            if entity_found:
                raise ValueError(f"Multiple entities found ({entity} and {entity})")
            else:
                entity_found = True
                queried_entity = entity
    if not entity_found:
        raise ValueError(f"No entity was found! {query} {context}")
    return queried_entity


def get_attribute(entity: Entity, queried_relation: Relation):
    """returns the value of the relation queried by the query, applied to the entity. Returns an error if no answer is found or if multiple answers are found."""
    answer_found = False
    answer = "NotFound!"
    for attribute in entity.attributes:
        if attribute.relation == queried_relation:
            if answer_found:
                raise ValueError(
                    f"Multiple answers found ({answer} and {entity.attributes[0].first_token})"
                )
            answer = attribute.first_token
            answer_found = True
    if not answer_found:
        raise ValueError(f"No answer was found! {queried_relation} {entity}")
    return answer


@define(auto_attribs=True)
class ContextQueryPrompt(CausalInput):
    context: List[Entity] = field()
    query: Query = field()
    model_input: str = field()
    causal_graph_input: Dict[str, Any] = field(init=False)
    tokenizer: Any = field(init=True, default=None)
    answer: str = field(init=False)

    def __attrs_post_init__(self):
        self.causal_graph_input = {"query": self.query, "context": self.context}
        self.answer = find_answer(self.query, self.context)


# define general causal graph

query = CausalGraph(name="query", output_type=Query, leaf=True)
context = CausalGraph(name="context", output_type=List, leaf=True)
CONTEXT_RETRIEVAL_CAUSAL_GRAPH = CausalGraph(
    name="output", output_type=str, f=find_answer, children=[query, context]
)

# define a fine-grained causal graph

query = CausalGraph(name="query", output_type=Query, leaf=True)
context = CausalGraph(name="context", output_type=List, leaf=True)

queried_relation = CausalGraph(
    name="queried_relation",
    output_type=Relation,
    children=[query],
    f=lambda query: query.queried_relation,
)
entity = CausalGraph(
    name="entity", output_type=Entity, children=[query, context], f=find_entity
)
FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH = CausalGraph(
    name="output", output_type=str, children=[entity, queried_relation], f=get_attribute
)
