# %%
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import (
    CausalAlignement,
    layer_span,
    check_alignement,
    InterchangeInterventionAccuracy,
    CausalInput,
)
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition
import torch
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset

# %%
a = CausalGraph(name="a", output_type=int, leaf=True)
b = CausalGraph(name="b", output_type=int, leaf=True)

f = lambda a, b: str(a + b)
c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

g = lambda c: c + "!"
d = CausalGraph(name="d", output_type=str, f=g, children=[c])

d.run({"a": 1, "b": 2}, fixed_inputs={"c": {"a": 5, "b": 10}}) == "15!"
# %%
a = CausalGraph(name="a", output_type=int, leaf=True)
b = CausalGraph(name="b", output_type=int, leaf=True)

f = lambda a, b: str(a + b)
c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

model = HookedTransformer.from_pretrained(model_name="pythia-70m")

position = WildPosition(position=9, label="test")

alig = CausalAlignement(
    causal_graph=c,
    model=model,
    mapping_tl={
        "a": layer_span(0, 1, position),
        "b": layer_span(2, 3, position),
        "c": layer_span(3, 6, position),
    },
)


nano_qa_dataset = NanoQADataset(
    nb_samples=30,
    tokenizer=model.tokenizer,
    querried_variables=["character_name", "city"],
)
# %%
dataset = []

for i in range(len(nano_qa_dataset)):
    dataset.append(
        CausalInput(
            causal_graph_input={"a": 1, "b": 2},
            model_input=nano_qa_dataset.prompts_text[i],
        )
    )

check_alignement(
    alig, model, c, dataset, InterchangeInterventionAccuracy, nb_inter=100, verbose=True
)

# %%
