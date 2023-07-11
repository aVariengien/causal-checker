# %%
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import (
    CausalAlignement,
    layer_span,
    check_alignement,
    evaluate_model,
    InterchangeInterventionAccuracy,
)
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import NanoQADataset
import torch
from causal_checker.retrieval import (
    CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
)
from causal_checker.datasets.nanoQA import create_nanoQA_retrieval_dataset
from functools import partial
import numpy as np
from causal_checker.hf_hooks import residual_steam_hook_fn, dummy_hook
from causal_checker.models import get_falcon_model, get_gpt2_model, get_pythia_model


def test_causal_alignement():
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


def test_wild_position():
    a = WildPosition(position=[5, 98, 2], label="test")
    assert a.positions_from_idx([1, 0, 2]) == [98, 5, 2]
    a = WildPosition(position=5, label="test")
    assert a.positions_from_idx([1, 0, 2]) == [5, 5, 5]

    a = torch.Tensor([3, 4])
    WildPosition(position=a, label="test")


def test_model_component():
    a = ModelComponent(position=4, layer=8, name="z", head=6, position_label="test")
    assert str(a) == "blocks.8.attn.hook_z.h6@test"

    t = torch.Tensor([3, 4])
    pos = WildPosition(position=t, label="test")
    ModelComponent(position=pos, layer=8, name="z", head=6)  # no label needed


def test_compute_alignement_nanoQA():
    model = HookedTransformer.from_pretrained(model_name="gpt2-small")

    nano_qa_dataset = NanoQADataset(
        nb_samples=100,
        tokenizer=model.tokenizer,  # type: ignore
        nb_variable_values=5,
        seed=42,
        querried_variables=["city", "character_name", "character_occupation"],
    )
    dataset = create_nanoQA_retrieval_dataset(nano_qa_dataset)

    end_position = WildPosition(position=nano_qa_dataset.word_idx["END"], label="END")

    MID_LAYER = 8
    alig = CausalAlignement(
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        model=model,
        mapping_tl={
            "query": layer_span(0, MID_LAYER, end_position),
            "context": layer_span(
                0, 0, position=WildPosition(position=0, label="dummy_position")
            ),
            "output": layer_span(MID_LAYER, model.cfg.n_layers, end_position),
        },
    )

    baseline, interchange_intervention_acc = check_alignement(
        alignement=alig,
        model=model,
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        dataset=dataset,
        compute_metric=partial(InterchangeInterventionAccuracy, compute_mean=False),
        variables_inter=["query"],
        nb_inter=100,
        batch_size=10,
        verbose=True,
        seed=42,
    )

    baseline_percentage, iia = np.count_nonzero(baseline), np.count_nonzero(
        interchange_intervention_acc
    )
    assert baseline_percentage > 0.8
    assert iia > 0.8

    # check against hf hooks
    model, tokenizer = get_gpt2_model("small", dtype=torch.float32)
    MID_LAYER = 8  #
    alig = CausalAlignement(
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        hook_type="hf",
        model=model,
        mapping_hf={
            "query": residual_steam_hook_fn(
                resid_layer=MID_LAYER, position=end_position
            ),
            "context": dummy_hook(),
            "output": dummy_hook(),
        },
    )
    hf_baseline, hf_interchange_intervention_acc = check_alignement(
        alignement=alig,
        model=model,
        causal_graph=CONTEXT_RETRIEVAL_CAUSAL_GRAPH,
        dataset=dataset,
        compute_metric=partial(InterchangeInterventionAccuracy, compute_mean=False),
        variables_inter=["query"],
        nb_inter=100,
        batch_size=10,
        verbose=True,
        seed=42,
        tokenizer=tokenizer,
    )
    assert hf_baseline == baseline
    assert hf_interchange_intervention_acc == interchange_intervention_acc

# %%
test_compute_alignement_nanoQA()
# %%
