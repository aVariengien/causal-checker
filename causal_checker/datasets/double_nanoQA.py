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
    Query,
    Attribute,
    Entity,
)

from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
import random as rd


NAME_GENDER = {
    "Michael": 0,
    "Christopher": 0,
    "Matthew": 0,
    "Ashley": 1,
    "Jessica": 1,
}

DOUBLE_NANO_QA_TEMPLATE = """

Here are two short stories. The first involve a male character while the second is about a female protagonist. Read them carefully and answer the questions below.

1. Female character
{FEMALE_STORY}

2. Male character
{MALE_STORY}

Answer the questions below, The answers should be concise and to the point.

Question: {question}

Answer: The answer is '"""


QUESTIONS = [
    # city
    {
        "question": "Where does the story of the {character} take place?",
        "answer_prefix": "The story of the {character} takes place in the city called",
        "querried_variable": "city",
    },
    {
        "question": "In which city is the plot of the {character} set?",
        "answer_prefix": "The plot of the {character}'s story takes place in the city of",
        "querried_variable": "city",
    },
    {
        "question": "Where is the story of the {character} located?",
        "answer_prefix": "The story of the {character} is located in a city named",
        "querried_variable": "city",
    },
    # occupation
    {
        "question": "What job does the {character} have?",
        "answer_prefix": "The {character} is a professional",
        "querried_variable": "character_occupation",
    },
    {
        "question": "In which profession is the {character} involved?",
        "answer_prefix": "The {character} is a professional",
        "querried_variable": "character_occupation",
    },
    {
        "question": "Which vocation does the {character} pursue?",
        "answer_prefix": "The {character} is a professional",
        "querried_variable": "character_occupation",
    },
]

GENDERED_CHARACTER_REFERENCES = ["protagonist", "character"]


def sample():
    pass


def create_double_nanoQA_retreival_dataset(
    tokenizer, nb_sample=100, prepend_space=True
) -> List[ContextQueryPrompt]:
    nano_qa_dataset = NanoQADataset(
        nb_samples=nb_sample * 3,
        tokenizer=tokenizer,
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )

    male_nanostories = []
    female_nanostories = []
    for s in nano_qa_dataset.nanostories:
        if NAME_GENDER[s["seed"]["character_name"]]:
            female_nanostories.append(s)
        else:
            male_nanostories.append(s)

    all_prompts = []

    for k in range(nb_sample):
        # generate the text of the prompt
        s_male = rd.choice(male_nanostories)
        s_female = rd.choice(female_nanostories)

        q = rd.choice(QUESTIONS)
        question, querried_var, answer_prefix = (
            q["question"],
            q["querried_variable"],
            q["answer_prefix"],
        )

        querried_gender = rd.choice(["male", "female"])
        reference = querried_gender + " " + rd.choice(GENDERED_CHARACTER_REFERENCES)
        question = question.format(character=reference)
        answer_prefix = answer_prefix.format(character=reference)

        double_nanoQA_text = DOUBLE_NANO_QA_TEMPLATE.format(
            FEMALE_STORY=s_female["story"],
            MALE_STORY=s_male["story"],
            question=question,
            answer_prefix=answer_prefix,
        )

        # generate the representation of the prompt
        entities = []
        for gender, story in zip(["male", "female"], [s_male, s_female]):
            attributes = [Attribute(value=gender, name="gender")]
            attributes += [
                Attribute(
                    value=" " * prepend_space + story["seed"][nar_var],
                    name=nar_var,
                    tokenizer=tokenizer,
                )
                for nar_var in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES
            ]
            entities.append(
                Entity(
                    name=" " * prepend_space + story["seed"]["character_name"],
                    attributes=attributes,
                    tokenizer=tokenizer,
                )
            )
        query = Query(
            queried_attribute=str(querried_var),
            filter_by=[Attribute(value=querried_gender, name="gender")],
        )

        prompt = ContextQueryPrompt(
            model_input=double_nanoQA_text,
            query=query,
            context=entities,
            tokenizer=nano_qa_dataset.tokenizer,
        )
        all_prompts.append(prompt)
    return all_prompts


# %%
