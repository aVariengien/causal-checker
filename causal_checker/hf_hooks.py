# %%
from transformers import GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM  # type: ignore
import torch
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal, Union
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
import numpy as np
from functools import partial


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    elif isinstance(model, GPT2LMHeadModel):
        return model.transformer.h
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.gpt_neox.layers
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def residual_steam_hook_fn(resid_layer: int, position: WildPosition):
    """resid_layer=0 is the embedding (not supported yet), 1 is after the first block, etc."""
    assert resid_layer > 0
    resid_layer -= 1 # offset by one to account for embedding layer
    def hook_fn(
        model,
        source_toks: torch.Tensor,
        target_toks: torch.Tensor,
        source_idx: np.ndarray,
        target_idx: np.ndarray,
    ):
        assert len(source_toks.shape) == 2
        assert len(target_toks.shape) == 2
        assert source_toks.shape[0] == target_toks.shape[0]
        assert len(list(source_idx)) == source_toks.shape[0]
        assert len(list(target_idx)) == target_toks.shape[0]

        blocks = get_blocks(model)
        position.positions_from_idx(list(target_idx))

        def read_output_hook(module, input, output, val):
            val["output"] = output[0][
                range(len(source_idx)), position.positions_from_idx(list(source_idx))
            ]

        def write_output_hook(module, input, output, val):
            output[0][
                range(len(source_idx)), position.positions_from_idx(list(target_idx)), :
            ] = val["output"][:, :]

        val = {}
        read_handle = blocks[resid_layer].register_forward_hook(
            partial(read_output_hook, val=val)
        )

        model(source_toks)
        read_handle.remove()
        write_handle = blocks[resid_layer].register_forward_hook(
            partial(write_output_hook, val=val)
        )
        return write_handle

    return hook_fn


def dummy_hook():
    def hook_fn(
        model,
        source_toks: torch.Tensor,
        target_toks: torch.Tensor,
        source_idx: np.ndarray,
        target_idx: np.ndarray,
    ):
        raise NotImplementedError("This is a dummy hook_fn")

    return hook_fn


# %%
