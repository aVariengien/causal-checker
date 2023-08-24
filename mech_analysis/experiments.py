# %%
import math
import random as rd
from functools import partial
from pprint import pprint
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from swap_graphs.core import WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    pprint_nanoqa_prompt,
    ALL_NAR_VAR,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import (
    check_tokenizer_nanoQA,
    print_performance_table,
)
from swap_graphs.utils import clean_gpu_mem, printw
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # Hooking utilities

torch.set_grad_enabled(False)


def get_dataset_pair(variables: List[str], dataset_size: int, model: HookedTransformer):
    c1_q1_dataset = NanoQADataset(
        nb_samples=dataset_size,
        name=f"C1 Q1 - ({' '.join(variables)})",
        tokenizer=model.tokenizer,  # type: ignore
        querried_variables=variables,
        seed=rd.randint(0, 100000),
        nb_variable_values=5,
    )

    c2_q2_dataset = NanoQADataset(
        nb_samples=dataset_size,
        name="C2 Q2" + str(variables),
        tokenizer=model.tokenizer,  # type: ignore
        querried_variables=variables,
        seed=rd.randint(0, 100000),
        nb_variable_values=5,
    )
    return c1_q1_dataset, c2_q2_dataset


def cache_final_state_heads(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
    model: HookedTransformer,
):
    """Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer."""
    assert isinstance(hook.name, str)
    assert "hook_z" in hook.name
    z_value = z[
        range(z.shape[0]), end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    layer = int(hook.name.split(".")[1])
    out_value = torch.einsum(
        "h z d, b h z -> b h d",
        model.W_O[layer],
        z_value,
    )
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_final_state(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
):
    assert isinstance(hook.name, str)
    assert (
        ("mlp_out" in hook.name)
        or ("hook_scale" in hook.name)
        or ("resid_pre" in hook.name)
        or ("resid_post" in hook.name)
    )
    out_value = z[
        range(z.shape[0]), end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_final_attn_probs(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    cache: Dict,
):
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    out_value = z[
        range(z.shape[0]), :, end_position.positions_from_idx(list(range(z.shape[0])))
    ]
    cache[hook.name + "-out-END"] = out_value.clone().detach().cpu()


def cache_value_norm_hook(z: torch.Tensor, hook: HookPoint, cache: Dict):
    assert isinstance(hook.name, str)
    assert "hook_v" in hook.name
    cache[hook.name + "-norm"] = (
        torch.norm(z, dim=-1, keepdim=True).clone().detach().cpu()
    )


value_filter = lambda x: "hook_v" in x
pattern_filter = lambda x: "pattern" in x
all_filter = lambda x: True
head_output_filter = lambda x: "hook_z" in x
mlp_output_filter = lambda x: "hook_mlp_out" in x
layer_norm_scale_filter = lambda x: "ln_final.hook_scale" in x


def fill_cache(
    cache,
    model,
    nano_qa_dataset: NanoQADataset,
    mid_layer: int,
    additional_hooks: List[Tuple[str, Callable]] = [],
    randomize: bool = False,
):
    if randomize:
        idx = torch.randint(0, len(nano_qa_dataset), (len(nano_qa_dataset),))
        end_position = WildPosition(
            position=torch.tensor(nano_qa_dataset.word_idx["END"])[idx], label="END"
        )
        inputs = nano_qa_dataset.prompts_tok[idx]
    else:
        end_position = WildPosition(
            position=nano_qa_dataset.word_idx["END"], label="END"
        )
        inputs = nano_qa_dataset.prompts_tok
    get_final_state_head = partial(
        cache_final_state_heads,
        cache=cache,
        end_position=end_position,
        model=model,
    )
    get_final_pattern = partial(
        cache_final_attn_probs,
        cache=cache,
        end_position=end_position,
    )
    get_final_state_general = partial(
        cache_final_state,
        cache=cache,
        end_position=end_position,
    )
    model.run_with_hooks(
        inputs,
        fwd_hooks=[
            (pattern_filter, get_final_pattern),
            (head_output_filter, get_final_state_head),
            (value_filter, partial(cache_value_norm_hook, cache=cache)),
            (mlp_output_filter, get_final_state_general),
            (layer_norm_scale_filter, get_final_state_general),
            (f"blocks.{mid_layer}.hook_resid_pre", get_final_state_general),
            (f"blocks.{model.cfg.n_layers-1}.hook_resid_post", get_final_state_general),
        ]
        + additional_hooks,
    )


# %%


def get_component_output(
    compo_type: str,
    layer: int,
    h: int,
    cache: Dict,
) -> torch.Tensor:
    if compo_type == "head":
        return cache["blocks.{}.attn.hook_z-out-END".format(layer)][:, h, :]
    elif compo_type == "mlp":
        return cache["blocks.{}.hook_mlp_out-out-END".format(layer)]
    raise ValueError("compo_type must be 'head' or 'mlp'")


def get_all_component_outputs(
    cache: Dict, model: HookedTransformer, dataset: NanoQADataset
):
    # get compoenent outputs
    x = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads + 1, len(dataset), model.cfg.d_model
    )  # all outputs
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads + 1):
            compo_type = "head" if h < model.cfg.n_heads else "mlp"
            x[layer, h] = get_component_output(
                compo_type, layer, h, cache
            )  # tensor of size (dataset_size,d_model)
    return x


def path_patching_logits(
    dataset: NanoQADataset,
    model: Any,
    cache: Dict,
    narrative_variable: str,
    corrupted_cache: Optional[Dict] = None,
    ref_narrative_variable: Optional[Union[str, List[str]]] = None,
    nb_ressamples: int = 1,
    probs: bool = False,
    log_probs: bool = False,
    return_ref: bool = False,
    cor_dataset: Optional[NanoQADataset] = None,
):
    if corrupted_cache is None:
        corrupted_cache = cache
    if cor_dataset is None:
        cor_dataset = dataset

    if isinstance(ref_narrative_variable, str):
        assert ref_narrative_variable in ALL_NAR_VAR
        ref_narrative_variable = [ref_narrative_variable]
    elif isinstance(ref_narrative_variable, list):
        for v in ref_narrative_variable:
            assert v in ALL_NAR_VAR
        assert len(set(ref_narrative_variable)) == len(ref_narrative_variable)
    elif ref_narrative_variable is None:
        ref_narrative_variable = ALL_NAR_VAR
    else:
        raise ValueError("ref_narrative_variable must be None, a str or a List[str]")

    ref_outputs = get_all_component_outputs(cache, model, dataset)
    cor_outputs = get_all_component_outputs(corrupted_cache, model, cor_dataset)

    def model_end(x):
        """Compute the final layer norm and unembedding"""
        res = model.ln_final(x.cuda().unsqueeze(0))
        return model.unembed(res).cpu().squeeze(0)

    last_resid = cache["blocks.31.hook_resid_post-out-END"]
    ref_logits = model_end(last_resid)
    cor_logits = model_end(corrupted_cache["blocks.31.hook_resid_post-out-END"])

    all_logits = torch.zeros(
        nb_ressamples,
        model.cfg.n_layers,
        model.cfg.n_heads + 1,
        len(dataset),
        model.cfg.d_vocab,
    )
    for k in range(nb_ressamples):
        rd_idx = torch.randint(0, len(cor_dataset), ref_outputs.shape[:-1])

        all_outputs_ressample = cor_outputs[
            torch.arange(ref_outputs.size(0)).unsqueeze(1).unsqueeze(2),
            torch.arange(ref_outputs.size(1)).unsqueeze(0).unsqueeze(2),
            rd_idx,
        ]
        for layer in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads + 1):
                ressample_last_resid = (
                    last_resid - ref_outputs[layer, h] + all_outputs_ressample[layer, h]
                )
                all_logits[k, layer, h] = model_end(ressample_last_resid)

    # compute metrics
    if probs:
        if log_probs:
            all_probs = torch.log_softmax(all_logits, dim=-1)
            ref_probs = torch.log_softmax(ref_logits, dim=-1)
            cor_probs = torch.log_softmax(cor_logits, dim=-1)
        else:
            all_probs = torch.softmax(all_logits, dim=-1)
            ref_probs = torch.softmax(ref_logits, dim=-1)
            cor_probs = torch.softmax(cor_logits, dim=-1)
        correct_all_probs = all_probs[
            :,
            :,
            :,
            range(len(dataset)),
            dataset.narrative_variables_token_id[narrative_variable],
        ]
        correct_ref_probs = ref_probs[
            range(len(dataset)),
            dataset.narrative_variables_token_id[narrative_variable],
        ]

        delta_probs = (
            correct_ref_probs.unsqueeze(0).unsqueeze(0).unsqueeze(0) - correct_all_probs
        )
        delta = delta_probs.mean(dim=0)
        ref = correct_ref_probs
        rd_ref = cor_probs
    else:

        def get_logit_diff(logits):
            correct_all_logits = logits[
                ...,
                range(len(dataset)),
                dataset.narrative_variables_token_id[narrative_variable],
            ]  # (..., dataset_size)

            all_indices = []

            for v in ref_narrative_variable:
                all_indices.append(
                    torch.tensor(dataset.narrative_variables_token_id[v]).unsqueeze(-1)
                )
            idx = torch.cat(all_indices, dim=-1)
            assert idx.shape[0] == len(dataset)
            assert idx.shape[1] == len(ref_narrative_variable)

            expanded_idx = idx.expand((*logits.shape[:-2], *idx.shape))
            baseline_logits = logits.gather(
                dim=-1, index=expanded_idx
            )  # (..., dataset_size, nb_var)
            mean_baseline_logits = baseline_logits.mean(dim=-1)  # (..., dataset_size)

            return correct_all_logits - mean_baseline_logits

        logit_diff_ref = get_logit_diff(ref_logits)
        rd_ref = get_logit_diff(cor_logits)
        logit_diff_all = get_logit_diff(all_logits)

        delta = (
            logit_diff_ref.unsqueeze(0).unsqueeze(0).unsqueeze(0) - logit_diff_all
        ).mean(dim=0)
        ref = logit_diff_ref
    if return_ref:
        return delta, ref, rd_ref
    else:
        return delta


def direct_logit_attribution(
    dataset: NanoQADataset,
    model: Any,
    cache: Dict,
    narrative_variable: str,
    ref_narrative_variable: Optional[str] = None,
    remove_mean: bool = True,
):
    """Compute the net contribution of each head and mlp to the final logit of the correct token.
    If ref_narrative_variable is set, it computes the attribution for the logit difference narrative_variable-ref_narrative_variable.
    If ref_narrative_variable is None, it computes the attribution for the logit of narrative_variable and removes the mean DLA of the logit of all other narrative variables.
    """
    # TODO: remove all of this and do the + - instead and run only the final layer norm. Compute the difference in log probs
    all_outputs = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads + 1, len(dataset), model.cfg.d_model
    )
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads + 1):
            compo_type = "head" if h < model.cfg.n_heads else "mlp"
            all_outputs[layer, h] = get_component_output(
                compo_type, layer, h, cache
            )  # tensor of size (dataset_size,d_model)
    if remove_mean:
        all_outputs -= all_outputs.mean(dim=2, keepdim=True)

    if ref_narrative_variable is None:
        unembeds = torch.cat(
            [
                model.W_U[
                    :, dataset.narrative_variables_token_id[narrative_variable][i]
                ].unsqueeze(0)
                for i in range(len(dataset))
            ],
            dim=0,  # (dataset_size, d_model)
        ).cpu()

        mean_var_unembeds = torch.zeros_like(unembeds)
        for var in ALL_NAR_VAR:
            mean_var_unembeds += torch.cat(
                [
                    model.W_U[
                        :, dataset.narrative_variables_token_id[var][i]
                    ].unsqueeze(0)
                    for i in range(len(dataset))
                ],
                dim=0,  # (dataset_size, d_model)
            ).cpu()
        mean_var_unembeds /= len(ALL_NAR_VAR)
        unembeds = (
            unembeds - mean_var_unembeds
        )  # remove the mean DLA of all other narrative variables
    else:
        unembeds = torch.cat(
            [
                (
                    model.W_U[
                        :, dataset.narrative_variables_token_id[narrative_variable][i]
                    ]
                    - model.W_U[
                        :,
                        dataset.narrative_variables_token_id[ref_narrative_variable][i],
                    ]
                ).unsqueeze(0)
                for i in range(len(dataset))
            ],
            dim=0,  # (dataset_size, d_model)
        ).cpu()

    layer_norms = cache["ln_final.hook_scale-out-END"].flatten()  # (dataset_size, 1)
    print(all_outputs.shape, unembeds.shape, layer_norms.shape)
    dla = torch.einsum(
        "l h b d, b d -> l h b", all_outputs, unembeds
    )  # (n_layers, n_heads+1, dataset_size)
    print(dla.shape)

    dla = torch.einsum(
        "l h b, b -> l h b", dla, 1 / layer_norms
    )  # (n_layers, n_heads+1, dataset_size)

    return dla


def get_attn_to_correct_tok(
    layer: int, h: int, dataset: NanoQADataset, cache, narrative_variable: str
):
    total_prob = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        correct_idx = dataset.narrative_variables_token_pos[narrative_variable][i]
        total_prob[i] = cache[f"blocks.{layer}.attn.hook_pattern-out-END"][i][h][
            correct_idx
        ].sum()

    return total_prob.cpu()

    # value_norm = cache[f"blocks.{layer}.attn.hook_v-norm"][dataset_idx, :, h]
    # return (pattern * value_norm).cpu().numpy()


def attention_to_nar_var_tok(
    dataset: NanoQADataset, model: Any, cache: Dict, narrative_variable: str
):
    attention = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, len(dataset)))
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            attention[layer, h] = get_attn_to_correct_tok(
                layer, h, dataset, cache, narrative_variable
            )
    return attention.numpy()


def get_str_toks(s: str, tokenizer: Any) -> List[str]:
    tok_ids = tokenizer(s)["input_ids"]
    l = [tokenizer.decode(i).replace(" ", "·") for i in tok_ids]
    return l


def steering_hook(
    z: torch.Tensor,
    hook: HookPoint,
    resid_source: torch.Tensor,
    end_position: Optional[WildPosition] = None,
):
    if end_position is None:
        assert z.shape[0] == 1
        z[0, -1, :] = resid_source.cuda()
    else:
        print("yeah", z.shape)
        z[
            range(z.shape[0]),
            end_position.positions_from_idx(list(range(z.shape[0]))),
            :,
        ] = resid_source.cuda()

    return z


def act_names(cache):
    for k, a in cache.items():
        print(k, a.shape)


def apply_steering(
    target_dataset: NanoQADataset,
    model: Any,
    cache_source: Dict,
    mid_layer: int,
):
    end_position_target = WildPosition(
        position=target_dataset.word_idx["END"], label="END"
    )
    model.add_hook(
        f"blocks.{mid_layer}.hook_resid_pre",
        partial(
            steering_hook,
            resid_source=cache_source[f"blocks.{mid_layer}.hook_resid_pre-out-END"],
            end_position=end_position_target,
        ),
    )


def attention_pattern_hook_precise(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    source_q_var_positions: List[List[int]],
    target_q_var_positions: List[List[int]],
):
    """
    Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer.
    Only changes the attentions pattern at the position of the first token of the target and source variable.
    """
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    layer = int(hook.name.split(".")[1])
    heads = [h for (l, h) in movers if l == layer]

    # [1, 32, 272, 272] OG pattern shape
    # hook_pattern-out-END torch.Size([100, 32, 289])
    source_pattern = source_cache[f"blocks.{layer}.attn.hook_pattern-out-END"]
    batch_size = z.shape[0]
    max_len = z.shape[2]
    end_pos = end_position.positions_from_idx(list(range(batch_size)))
    for h in heads:
        for i in range(batch_size):
            edit_pos = target_q_var_positions[i] + source_q_var_positions[i]
            if i == 0:
                print(f"target {h}")
                print(z[i, h, end_pos[i], target_q_var_positions[i]])
                print(source_pattern[i, h, target_q_var_positions[i]])
                print("source")
                print(z[i, h, end_pos[i], source_q_var_positions[i]])
                print(source_pattern[i, h, source_q_var_positions[i]])
            z[i, h, end_pos[i], edit_pos] = source_pattern[i, h, edit_pos].cuda()

    return z


def attention_pattern_hook_all_seq(
    z: torch.Tensor,
    hook: HookPoint,
    end_position: WildPosition,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    end_of_story_pos: List[int],
):
    """Extract the output of the end at the end position and multiply by the ouput matrix of the attention layer."""
    assert isinstance(hook.name, str)
    assert "pattern" in hook.name
    layer = int(hook.name.split(".")[1])
    heads = [h for (l, h) in movers if l == layer]

    # [1, 32, 272, 272] OG pattern shape
    # hook_pattern-out-END torch.Size([100, 32, 289])
    source_pattern = source_cache[f"blocks.{layer}.attn.hook_pattern-out-END"]
    batch_size = z.shape[0]
    max_len = z.shape[2]
    for h in heads:
        for i in range(batch_size):
            z[
                range(batch_size),
                h,
                end_position.positions_from_idx(list(range(batch_size))),
                : end_of_story_pos[i],
            ] = source_pattern[:, h, : end_of_story_pos[i]].cuda()

            norm_factor = z[
                range(batch_size),
                h,
                end_position.positions_from_idx(list(range(batch_size))),
            ].sum(dim=-1, keepdim=True)
            z[
                range(batch_size),
                h,
                end_position.positions_from_idx(list(range(batch_size))),
                :,
            ] /= norm_factor

    return z


pattern_filter = lambda x: "pattern" in x


def apply_attention_patch(
    model: HookedTransformer,
    movers: List[Tuple[int, int]],
    source_cache: Dict,
    target_dataset: NanoQADataset,
    source_dataset: NanoQADataset,
    mode: Literal["precise", "all_seq"] = "precise",
):
    assert mode in ["precise", "all_seq"]
    end_position_target = WildPosition(
        position=target_dataset.word_idx["END"], label="END"
    )

    for i in range(len(target_dataset)):
        assert (
            target_dataset.nanostory_end_token_pos[i]
            == source_dataset.nanostory_end_token_pos[i]
        )

    if mode == "all_seq":
        hook_fn = partial(
            attention_pattern_hook_all_seq,
            movers=movers,
            source_cache=source_cache,
            end_position=end_position_target,
            end_of_story_pos=target_dataset.nanostory_end_token_pos,
        )
    else:
        assert source_dataset is not None
        hook_fn = partial(
            attention_pattern_hook_precise,
            movers=movers,
            source_cache=source_cache,
            end_position=end_position_target,
            source_q_var_positions=source_dataset.narrative_variables_token_pos[
                "correct"
            ],
            target_q_var_positions=target_dataset.narrative_variables_token_pos[
                "correct"
            ],
        )

    model.add_hook(pattern_filter, hook_fn)


# %%
