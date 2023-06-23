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
                tokenizer=tokenizer,
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


NEWS_ARTICLE_ENGLISH = """
Title: "Climate Change: The Unsung Heroes"

In an era defined by increasing global temperatures and extreme weather events, the fight against climate change continues on many fronts. While prominent environmentalists and politicians often claim the limelight, behind the scenes, countless unsung heroes have dedicated their lives to combating climate change. This article aims to spotlight the work of these individuals.

At the forefront is M. {NAME1}, a marine biologist who has developed an innovative method for promoting coral reef growth. Given that coral reefs act as carbon sinks, absorbing and storing CO2 from the atmosphere, M. {NAME1}'s work has significant implications for climate mitigation. Despite facing numerous hurdles, M. {NAME1} has consistently pushed forward, driven by an unwavering commitment to oceanic health.

Next, we turn to M. {NAME2}, a climate economist from a small town who has successfully devised a market-based solution to curb industrial carbon emissions. By developing a novel carbon pricing model, M. {NAME2} has enabled a tangible shift toward greener industrial practices. The model has been adopted in several countries, resulting in significant reductions in CO2 emissions. Yet, despite these successes, M. {NAME2}'s work often flies under the mainstream media radar.

Another unsung hero in the climate change battle is M. {NAME3}, a young agricultural scientist pioneering a line of genetically modified crops that can thrive in drought conditions. With changing rainfall patterns threatening food security worldwide, M. {NAME3}'s work is of immense global relevance. However, due to controversy surrounding genetically modified organisms, the contributions of scientists like M. {NAME3} often go unnoticed.

Additionally, the story of M. {NAME4} is worth mentioning. An urban planner by profession, M. {NAME4} has been instrumental in designing green cities with a minimal carbon footprint. By integrating renewable energy sources, promoting public transportation, and creating more green spaces, M. {NAME4} has redefined urban living. While the aesthetics of these cities often capture public attention, the visionary behind them, M. {NAME4}, remains relatively unknown.

Lastly, we have M. {NAME5}, a grassroots activist working tirelessly to protect and restore the forests in her community. M. {NAME5} has mobilized local communities to halt deforestation and engage in extensive tree-planting initiatives. While large-scale afforestation projects often get global recognition, the efforts of community-level heroes like M. {NAME5} remain largely unsung.

The fight against climate change is not a single battle, but a war waged on multiple fronts. Every victory counts, no matter how small. So, as we continue this struggle, let's not forget to appreciate and honor the unsung heroes like M. {NAME1}, M. {NAME2}, M. {NAME3}, M. {NAME4}, and M. {NAME5} who, away from the spotlight, are making a world of difference."""


SENTENCES = [
    ("À l'avant-garde se trouve", "NAME1"),
    ("Ensuite, nous nous tournons vers", "NAME2"),
    (
        "Un autre héros méconnu dans la lutte contre le changement climatique est",
        "NAME3",
    ),
    ("De plus, l'histoire de", "NAME4"),
    ("Enfin, nous avons", "NAME5"),
]

NAMES = {
    "NAME1": ["Smith", "Johnson", "Williams", "Brown", "Jones"],
    "NAME2": ["Garcia", "Miller", "Davis", "Rodriguez", "Martinez"],
    "NAME3": ["Hernandez", "Lopez", "Gonzalez", "Perez", "Wilson"],
    "NAME4": ["Anderson", "Thomas", "Taylor", "Moore", "Jackson"],
    "NAME5": ["Martin", "Lee", "Walker", "Harris", "Thompson"],
}

TRANSLATION_TEMPLATE = """<|endoftext|>
English:
{ENGLISH_ARTICLE}

French:
[...]
{FRENCH_SENTENCE} M."""


def make_translation_prompt(tokenizer):
    names = {k: rd.choice(v) for k, v in NAMES.items()}
    sentence, querried_name = rd.choice(SENTENCES)
    english_article = NEWS_ARTICLE_ENGLISH.format(**names)
    translation_prompt = TRANSLATION_TEMPLATE.format(
        ENGLISH_ARTICLE=english_article, FRENCH_SENTENCE=sentence
    )

    entities = []
    for name_idx, name in names.items():
        entities.append(
            Entity(
                name=" " + name,
                attributes=[Attribute(value=name_idx, relation=Relation("name_order"))],
                tokenizer=tokenizer,
            )
        )
    query = Query(
        querried_relation=Relation("name"),
        filter_by=[Attribute(value=querried_name, relation=Relation("name_order"))],
    )
    prompt = ContextQueryPrompt(
        model_input=translation_prompt,
        query=query,
        context=entities,
        tokenizer=tokenizer,
    )
    return prompt


def create_translation_retrieval_dataset(
    nb_sample=100, tokenizer=None
) -> List[ContextQueryPrompt]:
    dataset = []
    for _ in range(nb_sample):
        dataset.append(make_translation_prompt(tokenizer))
    check_tokenisation(dataset)
    return dataset


# %%
