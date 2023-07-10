# %%
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
    OperationDataset,
)
from causal_checker.datasets.dataset_utils import gen_dataset_family

from swap_graphs.datasets.nano_qa.narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
import random as rd
from functools import partial

## CODE QUESTIONS

CODE_TEMPLATES = {
    "geometry": """
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
{Polygon} = Polygon([Point(0, 0), Point(1, 0), Point(0, 1)])

# Create a rectangle
{Rectangle} = Rectangle(Point(2, 3), Point(6, 5))

# Create a circle
{Circle} = Circle(Point(0, 0), 5)
""",
    "animals": """
class Dog:
    def __init__(self, name: str, breed: str, age: int):
        self.name = name
        self.breed = breed
        self.age = age

    def bark(self) -> None:
        print(f"{{self.name}} is barking!")

    def eat(self) -> None:
        print(f"{{self.name}} is eating.")

    def sleep(self) -> None:
        print(f"{{self.name}} is sleeping.")

class Cat:
    def __init__(self, name: str, color: str, weight: float):
        self.name = name
        self.color = color
        self.weight = weight

    def meow(self) -> None:
        print(f"{{self.name}} says meow!")

    def purr(self) -> None:
        print(f"{{self.name}} is purring.")

    def scratch(self) -> None:
        print(f"{{self.name}} is scratching.")


class Bird:
    def __init__(self, name: str, species: str, can_fly: bool):
        self.name = name
        self.species = species
        self.can_fly = can_fly

    def chirp(self) -> None:
        print(f"{{self.name}} is chirping!")

    def fly(self) -> None:
        if self.can_fly:
            print(f"{{self.name}} is flying.")
        else:
            print(f"{{self.name}} cannot fly.")

    def build_nest(self) -> None:
        print(f"{{self.name}} is building a nest.")


def train_animal(animal: Dog) -> None:
    # Perform some training actions specific to dogs
    animal.bark()
    animal.eat()
    animal.sleep()


def play_with_animal(animal: Cat) -> None:
    # Perform some play actions specific to cats
    animal.meow()
    animal.purr()
    animal.scratch()


def observe_animal(animal: Bird) -> None:
    # Perform some observation actions specific to birds
    animal.chirp()
    animal.fly()
    animal.build_nest()


# Testing the class definitions

{Dog} = Dog("Buddy", "Golden Retriever", 3)

{Bird} = Bird("Polly", "Parrot", True)

{Cat} = Cat("Whiskers", "Tabby", 5)

# Testing the functions

""",
    "banking": """
class BankAccount:
    def __init__(self, account_number: str, owner_name: str, balance: float = 0.0):
        self.account_number = account_number
        self.owner_name = owner_name
        self.balance = balance

    def deposit(self, amount: float) -> None:
        self.balance += amount
        print(f"Deposited {{amount}} into account {{self.account_number}}. New balance: {{self.balance}}")

    def withdraw(self, amount: float) -> None:
        if self.balance >= amount:
            self.balance -= amount
            print(f"Withdrew {{amount}} from account {{self.account_number}}. New balance: {{self.balance}}")
        else:
            print("Insufficient funds.")

    def check_balance(self) -> float:
        return self.balance


class CheckingAccount:
    def __init__(self, account_number: str, owner_name: str, balance: float = 0.0):
        self.bank_account = BankAccount(account_number, owner_name, balance)

    def deposit(self, amount: float) -> None:
        self.bank_account.deposit(amount)

    def withdraw(self, amount: float) -> None:
        self.bank_account.withdraw(amount)

    def check_balance(self) -> float:
        return self.bank_account.check_balance()


class SavingsAccount:
    def __init__(self, account_number: str, owner_name: str, interest_rate: float, balance: float = 0.0):
        self.bank_account = BankAccount(account_number, owner_name, balance)
        self.interest_rate = interest_rate

    def deposit(self, amount: float) -> None:
        self.bank_account.deposit(amount)

    def withdraw(self, amount: float) -> None:
        self.bank_account.withdraw(amount)

    def check_balance(self) -> float:
        return self.bank_account.check_balance()

    def calculate_interest(self) -> float:
        return self.bank_account.balance * self.interest_rate


class CreditCardAccount:
    def __init__(self, account_number: str, owner_name: str, credit_limit: float, balance: float = 0.0):
        self.bank_account = BankAccount(account_number, owner_name, balance)
        self.credit_limit = credit_limit

    def make_purchase(self, amount: float) -> None:
        if self.bank_account.balance + amount <= self.credit_limit:
            self.bank_account.balance += amount
            print(f"Made a purchase of {{amount}} with credit card {{self.bank_account.account_number}}. "
                  f"New balance: {{self.bank_account.balance}}")
        else:
            print("Purchase declined. Exceeds credit limit.")

    def make_payment(self, amount: float) -> None:
        self.bank_account.balance -= amount
        print(f"Made a payment of {{amount}} towards credit card {{self.bank_account.account_number}}. "
              f"New balance: {{self.bank_account.balance}}")
              
def get_account_details(account: CheckingAccount) -> dict:
    return {{
        "Account Number": account.bank_account.account_number,
        "Owner Name": account.bank_account.owner_name,
        "Balance": account.bank_account.balance
    }}

def is_credit_limit_exceeded(account: CreditCardAccount) -> bool:
    return account.bank_account.balance > account.credit_limit

def get_account_summary(account: BankAccount) -> str:
    summary = f"Account Number: {{account.account_number}}\n"
    summary += f"Owner Name: {{account.owner_name}}\n"
    summary += f"Balance: {{account.balance}}"
    return summary


def calculate_interest_earned(account: SavingsAccount) -> float:
    return account.calculate_interest()
    
# define accounts

{CheckingAccount} = CheckingAccount("123456789", "John Doe", 1000.0)
{CheckingAccount}.deposit(1000.0)
{CheckingAccount}.withdraw(500.0)


{CreditCardAccount} = CreditCardAccount("246813579", "Alice Johnson", 5000.0)
{CreditCardAccount}.make_purchase(1000.0)
{CreditCardAccount}.make_purchase(5000.0)

{SavingsAccount} = SavingsAccount("0987654321", "Jane Smith", 0.05)
{SavingsAccount}.deposit(2000.0)
{SavingsAccount}.withdraw(1000.0)

# Test the accounts function. 

""",
}

CODE_QUERRIES = {
    "geometry": [
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
    ],
    "animals": [
        (
            """
            # Train the dog
            train_animal(""",
            "Dog",
        ),
        (
            """
            # Play with the cat
            play_with_animal(""",
            "Cat",
        ),
        (
            """
            # Observe the bird
            observe_animal(""",
            "Bird",
        ),
    ],
    "banking": [
        (
            """
            # Print the details of a checking account
            print(get_account_details(""",
            "CheckingAccount",
        ),
        (
            """
            # Check if the credit card limit is exceeded
            print(is_credit_limit_exceeded(""",
            "CreditCardAccount",
        ),
        (
            """
            # Print the earned interest from the savings account
            print(calculate_interest_earned(""",
            "SavingsAccount",
        ),
    ],
}


def gen_code_prompt(tokenizer, dataset_name: str) -> ContextQueryPrompt:
    variables = [chr(rd.randint(0, 25) + 65) for _ in range(3)]
    # ensure that variables are all distinct
    while len(set(variables)) != 3:
        variables = [chr(rd.randint(0, 25) + 65) for _ in range(3)]

    types = [t for q, t in CODE_QUERRIES[dataset_name]]
    variables_dict = {types[i]: variables[i] for i in range(len(variables))}

    ctx_text = CODE_TEMPLATES[dataset_name].format(**variables_dict)
    query_text, type_querried = rd.choice(CODE_QUERRIES[dataset_name])

    entities = []

    for i, v in enumerate(variables):
        entities.append(
            Entity(
                name=v,
                attributes=[
                    Attribute(value=types[i], name=str("type"), to_tokenize=False)
                ],
                tokenizer=tokenizer,
                tokenize_name=True,
            )
        )
    query = Query(
        queried_attribute=str("name"),
        filter_by=[Attribute(value=type_querried, name=str("type"))],
    )
    prompt = ContextQueryPrompt(
        model_input=ctx_text + query_text,
        query=query,
        context=entities,
        tokenizer=tokenizer,
    )
    return prompt


def create_code_type_retrieval_dataset(
    nb_sample=100, tokenizer=None, dataset_names: Optional[List[str]] = None
) -> List[OperationDataset]:
    if dataset_names is None:
        dataset_names = list(CODE_TEMPLATES.keys())
    return gen_dataset_family(
        partial(gen_code_prompt, tokenizer),
        dataset_prefix_name="code_type_retrieval",
        dataset_names=dataset_names,
        nb_sample=nb_sample,
    )
