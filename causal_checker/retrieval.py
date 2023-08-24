# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.core import WildPosition

import torch
from causal_checker.utils import get_first_token
import random as rd


@define
class CausalInput:
    causal_graph_input: Dict[str, Any] = field()
    model_input: str = field()


@define
class Attribute:
    value: str = field()
    first_token: str = field(init=False)
    name: str = field()
    tokenizer: Any = field(init=True, default=None)
    to_tokenize: bool = field(default=True)

    def __attrs_post_init__(self):
        if not self.to_tokenize:
            self.tokenizer = None
        self.first_token = get_first_token(self.tokenizer, self.value)

    def __hash__(self):
        return hash(self.value + "---" + self.name)

    def __eq__(self, other):
        assert isinstance(other, Attribute)
        return self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.name}={self.value} (first_tok={self.first_token}))"

    def __repr__(self) -> str:
        return self.__str__()


@define
class Entity:
    name: str = field()
    attributes: List[Attribute] = field(factory=list)
    tokenizer: Any = field(init=True, default=None)
    tokenize_name: bool = field(default=False)

    def __attrs_post_init__(self):
        self.attributes.append(
            Attribute(
                name="name",
                value=self.name,
                tokenizer=self.tokenizer,
                to_tokenize=self.tokenize_name,
            )
        )
        self.attributes.append(Attribute(name="exists", value="yes", to_tokenize=False))
        self.propagate_tokenizer()

    def propagate_tokenizer(self):
        for a in self.attributes:
            assert isinstance(a, Attribute)
            assert isinstance(a.name, str)
            assert isinstance(a.value, str)
            if a.to_tokenize:
                if a.tokenizer is None:
                    a.tokenizer = self.tokenizer
                    a.first_token = get_first_token(self.tokenizer, a.value)
                assert a.first_token == get_first_token(
                    self.tokenizer, a.value
                ), "Incompatible tokenizer for entity and attribute"

    def __hash__(self):
        return hash(self.name)


@define
class Query:
    """A query is asking for the value of an attribute defined by the relation `query`. The entities find are the one that have all the attributes `filter_by`."""

    queried_attribute: str = field(default="name")
    filter_by: List[Attribute] = field(factory=list)

    def __attrs_post_init__(self):
        assert isinstance(self.queried_attribute, str)
        assert isinstance(self.filter_by, list)
        self.filter_by.append(Attribute(name="exists", value="yes"))


def find_answer(query: Query, context: List[Entity], nil: str = "nil"):
    """Apply the query on the context. Retruns the first token of the answer. Returns an error if no answer is found or if multiple answers are found."""
    answer_found = False
    answer = "NotFound!"
    for entity in context:
        if set(query.filter_by).issubset(
            set(entity.attributes)
        ):  # all attributes in filter_by are in entity.attributes
            for attribute in entity.attributes:
                if attribute.name == query.queried_attribute:
                    if answer_found:
                        raise ValueError(
                            f"Multiple answers found ({answer} and {entity.attributes[0].first_token})"
                        )
                    answer = attribute.first_token
                    answer_found = True
    if not answer_found:
        raise ValueError(f"No answer was found! {query} {context}")
    return answer


def find_entity(query_entity_filter: List[Attribute], context: List[Entity]):
    """Returns the enitity that matches the filter_by attribute of the query. Returns an error if no answer is found or if multiple answers are found."""
    entity_found = False
    queried_entity = None
    for entity in context:
        if set(query_entity_filter).issubset(
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


def get_attribute(entity: Entity, queried_attribute: str):
    """returns the value of the relation queried by the query, applied to the entity. Returns an error if no answer is found or if multiple answers are found."""
    answer_found = False
    answer = "NotFound!"
    for attribute in entity.attributes:
        if attribute.name == queried_attribute:
            if answer_found:
                raise ValueError(
                    f"Multiple answers found ({answer} and {entity.attributes[0].first_token})"
                )
            answer = attribute.first_token
            answer_found = True
    if not answer_found:
        raise ValueError(f"No answer was found! {queried_attribute} {entity}")
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
        self.propagate_tokenizer()
        self.causal_graph_input = {"query": self.query, "context": self.context}
        self.answer = find_answer(self.query, self.context)
        if self.model_input[:13] != "<|endoftext|>":
            self.model_input = "<|endoftext|>" + self.model_input
        self.check_tokenisation_incoherence()
        self.check_tokenisation_fit()

    def propagate_tokenizer(self):
        for entity in self.context:
            if entity.tokenizer is None:
                entity.tokenizer = self.tokenizer
                entity.propagate_tokenizer()

    def check_tokenisation_incoherence(self):
        """Verify that if the flag `tokenize_name` is on on at least one entity, then the query is asking for the name."""
        for entity in self.context:
            if entity.tokenize_name:
                if self.query.queried_attribute != "name":
                    raise ValueError(
                        f"If an entity has the flag `tokenize_name` on, then the query must ask for the name. {entity} {self.query}"
                    )

    def check_tokenisation_fit(self):
        """Assert that tokenize(prompt+answer) = tokenize(prompt) + tokenize(answer)."""
        prompt = self.model_input
        answer = self.answer
        prompt_tokenized = self.tokenizer.tokenize(prompt)
        answer_tokenized = self.tokenizer.tokenize(answer)
        prompt_answer_tokenized = self.tokenizer.tokenize(prompt + answer)
        if prompt_tokenized + answer_tokenized != prompt_answer_tokenized:
            raise ValueError(
                f"Tokenization doesn't fit! {prompt} {answer} | {prompt_tokenized} + {answer_tokenized} != {prompt_answer_tokenized}"
            )


def detect_first_token_collision(entities: List[Entity]):
    """Detect if two entities have an attribute of different value but with the the same first token. This would create ambiguity in the model input."""
    first_tokens = []
    all_attributes = {}
    for entity in entities:
        for attribute in entity.attributes:
            if attribute.first_token not in all_attributes:
                all_attributes[attribute.first_token] = attribute.value
            else:
                if (
                    all_attributes[attribute.first_token] != attribute.value
                ):  # different value but same first token
                    return (
                        attribute.value,
                        all_attributes[attribute.first_token],
                        attribute.first_token,
                    )
    return None


@define
class OperationDataset:
    """A dataset of operations. Each operation is a ContextQueryPrompt."""

    operations: List[ContextQueryPrompt] = field()
    name: str = field()
    check_for_collision: bool = field(default=True)

    def __attrs_post_init__(self):
        if self.check_for_collision:
            self.check_collision()
        self.compute_random_guess_accuracy()

    def check_collision(self):
        all_entities = []
        for prompt in self.operations:
            for entity in prompt.context:
                all_entities.append(entity)
        collision = detect_first_token_collision(all_entities)
        if collision is not None:
            val1, val2, first_tok = collision
            raise ValueError(
                f"Two different attribute values gives the same first token! {first_tok} {val1} {val2}"
            )

    def get_end_position(self) -> WildPosition:
        """Return the position of the last token of the model input."""
        end_positions = []
        for prompt in self.operations:
            end_positions.append(
                len(prompt.tokenizer(prompt.model_input)["input_ids"]) - 1
            )
        return WildPosition(position=end_positions, label="END")

    def compute_random_guess_accuracy(self) -> float:
        """Compute the accuracy of a random guess. Random guess is the accuracy of a model that is blind to the
        query and answer execute a random query on the context."""
        queries = [operation.query for operation in self.operations]
        contexts = [operation.context for operation in self.operations]
        nb_samples = 1000
        accuracy = 0
        for k in range(nb_samples):
            i = rd.randint(0, len(self.operations) - 1)
            real_answer = self.operations[i].answer

            rd_query = rd.choice(queries)
            rd_answer = find_answer(
                rd_query, contexts[i]
            )  # execute a random query on the context
            accuracy += int(rd_answer == real_answer)

        return accuracy / nb_samples

    def __getitem__(self, index):
        return self.operations[index]

    def __len__(self):
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)


# define general causal graph

query = CausalGraph(name="query", output_type=Query, leaf=True)
context = CausalGraph(name="context", output_type=List, leaf=True)
nil = CausalGraph(
    name="nil", output_type=str, f=lambda context: "nil", children=[context]
)  # dummy variable that has no effect
CONTEXT_RETRIEVAL_CAUSAL_GRAPH = CausalGraph(
    name="output", output_type=str, f=find_answer, children=[query, context, nil]
)


# define a fine-grained causal graph

query = CausalGraph(name="query", output_type=Query, leaf=True)
context = CausalGraph(name="context", output_type=List, leaf=True)

queried_attribute = CausalGraph(
    name="queried_attribute",
    output_type=str,
    children=[query],
    f=lambda query: query.queried_attribute,
)

query_entity_filter = CausalGraph(
    name="query_entity_filter",
    output_type=list,
    children=[query],
    f=lambda query: query.filter_by,
)

entity = CausalGraph(
    name="entity",
    output_type=Entity,
    children=[query_entity_filter, context],
    f=find_entity,
)

FINE_GRAINED_CONTEXT_RETRIEVAL_CAUSAL_GRAPH = CausalGraph(
    name="output",
    output_type=str,
    children=[entity, queried_attribute],
    f=get_attribute,
)

# %%
