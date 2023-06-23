# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import torch
from causal_checker.retrieval import (
    CausalInput,
    ContextQueryPrompt,
    Relation,
    Query,
    Attribute,
    Entity,
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
import random as rd


def check_tokenisation(dataset: List[ContextQueryPrompt]):
    all_attributes = []
    all_first_tokens = []
    for prompt in dataset:
        for entity in prompt.context:
            for attribute in entity.attributes:
                all_attributes.append(attribute.value)
                all_first_tokens.append(attribute.first_token)
    assert len(set(all_attributes)) == len(
        set(all_first_tokens)
    ), "Some first tokens are the same for different attributes!"


def create_nanoQA_retrieval_dataset(
    nano_qa_dataset: NanoQADataset,
) -> List[ContextQueryPrompt]:
    dataset = []

    for i in range(len(nano_qa_dataset)):
        q_var = nano_qa_dataset.questions[i]["querried_variable"]
        query = Query(
            querried_relation=Relation("name"),
            filter_by=[Attribute(value=q_var, relation=Relation("narrative_variable"))],
        )

        entities = []
        for nar_var in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES:
            entity_name = " " + nano_qa_dataset.nanostories[i]["seed"][nar_var]
            attr = Attribute(value=nar_var, relation=Relation("narrative_variable"))
            entities.append(
                Entity(
                    name=entity_name,
                    attributes=[attr],
                    tokenizer=nano_qa_dataset.tokenizer,
                )
            )

        prompt = ContextQueryPrompt(
            model_input=nano_qa_dataset.prompts_text[i],
            query=query,
            context=entities,
            tokenizer=nano_qa_dataset.tokenizer,
        )
        assert (
            prompt.answer == nano_qa_dataset.answer_first_token_texts[i]
        ), f"NanoQA and ContextQueryPrompt answers are different! {prompt.answer} vs {nano_qa_dataset.answer_first_token_texts[i]}"

        dataset.append(prompt)

    check_tokenisation(dataset)
    return dataset


## CODE QUESTIONS

CODE = """
from typing import List
from math import pi

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, bottom_left: Point, top_right: Point) -> None:
        self.bottom_left = bottom_left
        self.top_right = top_right
        
class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

class Polygon:
    def __init__(self, points: List[Point]) -> None:
        self.points = points

def calculate_area(rectangle: Rectangle) -> float:
    height = rectangle.top_right.y - rectangle.bottom_left.y
    width = rectangle.top_right.x - rectangle.bottom_left.x
    return height * width

def calculate_center(rectangle: Rectangle) -> Point:
    center_x = (rectangle.bottom_left.x + rectangle.top_right.x) / 2
    center_y = (rectangle.bottom_left.y + rectangle.top_right.y) / 2
    return Point(center_x, center_y)

def calculate_distance(point1: Point, point2: Point) -> float:
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5

def calculate_circumference(circle: Circle) -> float:
    return 2 * pi * circle.radius

def calculate_circle_area(circle: Circle) -> float:
    return pi * (circle.radius ** 2)

def calculate_perimeter(polygon: Polygon) -> float:
    perimeter = 0
    points = polygon.points + [polygon.points[0]]  # Add the first point at the end for a closed shape
    for i in range(len(points) - 1):
        perimeter += calculate_distance(points[i], points[i + 1])
    return perimeter

# Create a polygon
{POLYGON} = Polygon([Point(0, 0), Point(1, 0), Point(0, 1)])

# Create a rectangle
{RECTANGLE} = Rectangle(Point(2, 3), Point(6, 5))

# Create a circle
{CIRCLE} = Circle(Point(0, 0), 5)
"""

CODE_QUERRIES = [
    (
        """
# Calculate area
print(calculate_area(""",
        "Rectangle",
    ),
    (
        """
# Calculate circumference
print(calculate_circumference(""",
        "Circle",
    ),
    (
        """
# Calculate perimeter
print(calculate_perimeter(""",
        "Polygon",
    ),
]


def gen_code_prompt(tokenizer):
    variables = [chr(rd.randint(0, 25) + 65) for _ in range(3)]
    # ensure that variables are all distinct
    while len(set(variables)) != 3:
        variables = [chr(rd.randint(0, 25) + 65) for _ in range(3)]

    ctx_text = CODE.format(
        POLYGON=variables[0], RECTANGLE=variables[1], CIRCLE=variables[2]
    )
    query_text, type_querried = rd.choice(CODE_QUERRIES)

    entities = []
    types = ["Polygon", "Rectangle", "Circle"]
    for i, v in enumerate(variables):
        entities.append(
            Entity(
                name=v,
                attributes=[Attribute(value=types[i], relation=Relation("type"))],
                tokenizer=None,
            )
        )
    query = Query(
        querried_relation=Relation("name"),
        filter_by=[Attribute(value=type_querried, relation=Relation("type"))],
    )
    prompt = ContextQueryPrompt(
        model_input=ctx_text + query_text,
        query=query,
        context=entities,
        tokenizer=tokenizer,
    )
    return prompt


def create_code_type_retrieval_dataset(
    nb_sample=100, tokenizer=None
) -> List[ContextQueryPrompt]:
    dataset = []
    for _ in range(nb_sample):
        dataset.append(gen_code_prompt(tokenizer))
    check_tokenisation(dataset)
    return dataset


def get_end_position(dataset: List[ContextQueryPrompt], tokenizer: Any) -> WildPosition:
    """Return the position of the last token of the model input."""
    end_positions = []
    for prompt in dataset:
        end_positions.append(len(tokenizer(prompt.model_input)["input_ids"]) - 1)
    return WildPosition(position=end_positions, label="END")

# %%
