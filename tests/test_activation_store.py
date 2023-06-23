# %%
from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest
from causal_checker.alignement import CausalAlignement, layer_span
from transformer_lens import HookedTransformer
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
import torch


def test_activation_store():
    """Ensure persistence of the activation cache in the hook functions"""
    model = HookedTransformer.from_pretrained(model_name="pythia-70m")
    dataset_tok = torch.randint(1000, 20000, (2,5))
    activation_store = ActivationStore(
        model=model,
        dataset=dataset_tok,
        listOfComponents=[],
        force_cache_all=False,
    )
    
    components = [
        ModelComponent(position=1, layer=4, name="mlp",  position_label="test"),
        ModelComponent(position=1, layer=0, name="mlp",  position_label="test"),
        ModelComponent(position=1, layer=4, name="mlp",  position_label="test")
    ]
    
    all_hooks = []
    for c in components:
        activation_store.change_component_list([c])
        hook_list = activation_store.getPatchingHooksByIdx(
                source_idx=[0],
                target_idx=[1],
                list_of_components=[c],
            )
        all_hooks += hook_list
    
    hook_fn1 = all_hooks[0][1]
    hook_fn2 = all_hooks[1][1]
    hook_fn3 = all_hooks[2][1]
    
    
    t1 = hook_fn1(torch.zeros(2, 5, model.cfg.d_model), None)
    t2 = hook_fn2(torch.zeros(2, 5, model.cfg.d_model), None)
    t3 = hook_fn3(torch.zeros(2, 5, model.cfg.d_model), None)
    
    assert torch.allclose(t1, t3)
    assert not torch.allclose(t1, t2)
    assert not torch.allclose(t3, t2)
    
# %%
model = HookedTransformer.from_pretrained(model_name="pythia-70m")
# %%
