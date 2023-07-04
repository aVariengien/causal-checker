# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal, Union
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import torch
from causal_checker.retrieval import CausalInput, ContextQueryPrompt
from causal_checker.hf_hooks import residual_steam_hook_fn

Dataset = List[CausalInput] | List[ContextQueryPrompt]
Metric = Callable[[Any, List[Any], torch.Tensor, Dataset, List[int]], Any]


@define
class CausalAlignement:
    mapping_tl: Dict[str, List[ModelComponent]] = field(default={})
    mapping_hf: Dict[str, Callable] = field(default={})
    hook_type: Literal["hf", "transformerlens"] = field(default="transformerlens")

    def __init__(self, causal_graph, model, **kwargs):
        self.__attrs_init__(**kwargs)  # type: ignore
        assert self.hook_type in [
            "hf",
            "transformerlens",
        ], "hook_type must be hf or transformerlens"
        if self.hook_type == "transformerlens":
            assert len(self.mapping_tl) > 0, "mapping_tl must be non empty"
            self.check_hook_names(model)
        elif self.hook_type == "hf":
            assert len(self.mapping_hf) > 0, "mapping_hf must be non empty"
            self.check_hook_hf(model)
        self.check_mapping_inputs(causal_graph)
        self.check_model_causality(causal_graph, model)

    def check_mapping_inputs(self, causal_graph):
        all_nodes_names = causal_graph.get_all_nodes_names()
        if self.hook_type == "transformerlens":
            mapping = self.mapping_tl
        else:
            mapping = self.mapping_hf
        assert (
            set(list(mapping.keys())) == all_nodes_names
        ), f"mapping keys {set(self.mapping_tl.keys())} should be equal to all nodes names {all_nodes_names}"

    def check_hook_names(self, model):
        for node_name, components in self.mapping_tl.items():
            assert (
                type(components) == list
            ), f"node={node_name}, {components} is not a list"
            for component in components:
                assert (
                    component.hook_name in model.hook_dict
                ), f"{component.hook_name} not in the model hook_dict"

    def check_hook_hf(self, model):
        for node_name, hook_fn in self.mapping_hf.items():
            assert callable(hook_fn), f"node={node_name}, {hook_fn} is not callable"

    def check_model_causality(self, causal_graph, model):
        """Check that the causal alignement respect the topological order of the model's computationnal graph (i.e. weak test of graph homomorphism)"""
        pass  # TODO


def layer_span(layer_start: int, layer_end: int, position: WildPosition):
    """Returns a list of ModelComponent corresponding to the span of layers from layer_start to layer_end (excluded) at the given position"""
    components = []
    if layer_start == 0:
        components.append(ModelComponent(name="embed", layer=0, position=position))
    for i in range(layer_start, layer_end):
        components.append(ModelComponent(name="attn", layer=i, position=position))
        components.append(ModelComponent(name="mlp", layer=i, position=position))
    return components


def hf_apply_hooks(
    model: HookedTransformer,
    dataset_tok: torch.Tensor,
    variables_inter: List[str],
    alignement: CausalAlignement,
    input_indices: np.ndarray,
    batch_size: int,
    nb_inter: int,
    verbose: bool,
) -> torch.Tensor:
    logits = []
    for batch in range(0, nb_inter, batch_size):
        if verbose:
            print(f"batch {batch}")
        target_idx = input_indices[
            batch : batch + batch_size, 0
        ]  # the indices of the ref inputs
        target_toks = dataset_tok[target_idx]

        all_handles = []
        for k, variable in enumerate(variables_inter):
            source_idx = input_indices[batch : batch + batch_size, k + 1]
            source_toks = dataset_tok[source_idx]
            handle = alignement.mapping_hf[variable](
                model,
                source_toks=source_toks,
                target_toks=target_toks,
                source_idx=source_idx,
                target_idx=target_idx,
            )
            all_handles.append(handle)
        logits.append(model(target_toks).logits)
        for handle in all_handles:
            handle.remove()
    all_batch_logits = torch.cat(logits, dim=0)
    return all_batch_logits


def tl_apply_hooks(
    model: HookedTransformer,
    dataset_tok: torch.Tensor,
    variables_inter: List[str],
    alignement: CausalAlignement,
    input_indices: np.ndarray,
    batch_size: int,
    nb_inter: int,
    verbose: bool,
) -> torch.Tensor:
    activation_store = ActivationStore(
        model=model,
        dataset=dataset_tok,
        listOfComponents=[],
        force_cache_all=False,
    )
    # run interchange on the model
    logits = []
    for batch in range(0, nb_inter, batch_size):
        if verbose:
            print(f"batch {batch}")
        target_idx = input_indices[
            batch : batch + batch_size, 0
        ]  # the indices of the ref inputs
        target_toks = dataset_tok[target_idx]

        all_hooks = []
        for k, variable in enumerate(variables_inter):
            source_idx = input_indices[batch : batch + batch_size, k + 1]
            activation_store.change_component_list(alignement.mapping_tl[variable])

            assert len(source_idx.shape) == 1 and len(target_idx.shape) == 1
            assert type(alignement.mapping_tl[variable]) == list
            hook_list = activation_store.getPatchingHooksByIdx(
                source_idx=list(source_idx),
                target_idx=list(target_idx),
                list_of_components=alignement.mapping_tl[variable],
            )
            all_hooks += hook_list
        batch_logits = model.run_with_hooks(target_toks, fwd_hooks=all_hooks)
        logits.append(batch_logits)
    all_batch_logits = torch.cat(logits, dim=0)
    return all_batch_logits


def batched_forward(
    model: Union[HookedTransformer, Any], batch_size: int, dataset_toks: torch.Tensor
):
    all_logits = []
    is_tl = isinstance(model, HookedTransformer)
    for b in range(0, dataset_toks.shape[0], batch_size):
        if is_tl:
            all_logits.append(model(dataset_toks[b : b + batch_size]))
        else:
            all_logits.append(model(dataset_toks[b : b + batch_size]).logits)
    return torch.cat(all_logits, dim=0)


def evaluate_model(
    dataset: Dataset,
    batch_size: int,
    model: HookedTransformer,
    causal_graph: CausalGraph,
    compute_metric: Metric,
    tokenizer: Any = None,
):
    if isinstance(model, HookedTransformer):
        tokenizer = model.tokenizer
    else:
        assert (
            tokenizer is not None
        ), "tokenizer must be provided if model is not a HookedTransformer"
    dataset_str = [i.model_input for i in dataset]
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"])  # type: ignore
    logits = batched_forward(model, batch_size, dataset_tok)
    cg_output = []
    for i in range(len(dataset)):
        cg_output.append(causal_graph.run(inputs=dataset[i].causal_graph_input))
    baseline_metric = compute_metric(
        tokenizer, cg_output, logits, dataset, list(range(len(dataset)))
    )
    return baseline_metric


def check_alignement(
    alignement: CausalAlignement,
    model: HookedTransformer,
    causal_graph: CausalGraph,
    dataset: Dataset,
    compute_metric: Metric,
    variables_inter: Optional[List[str]] = None,
    nb_inter: int = 100,
    batch_size: int = 10,
    verbose: bool = False,
    tokenizer: Any = None,
    seed: Optional[int] = None,
    eval_baseline: bool = True,
) -> Any:
    """Check the alignement between the high-level causal graph and the low-level graph (the model) on the given dataset by performing interchange intervention.
    * variables_inter is the list of variables names from the causal graph to be intervened on. By default, all variables are intervened on.
    * nb_inter is the number of intervention to perform
    """
    np.random.seed(seed)
    if variables_inter is None:
        variables_inter = list(alignement.mapping_tl.keys())
        for v in variables_inter:
            assert (
                v in causal_graph.get_all_nodes_names()
            ), f"{v} not in the causal graph"
    nb_variable_inter = len(variables_inter)

    if isinstance(model, HookedTransformer):
        tokenizer = model.tokenizer
    else:
        assert (
            tokenizer is not None
        ), "tokenizer must be provided if model is not a HookedTransformer"

    dataset_str = [i.model_input for i in dataset]
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"])  # type: ignore

    if eval_baseline:
        baseline_metric = evaluate_model(
            dataset,
            batch_size,
            model,
            causal_graph,
            compute_metric,
            tokenizer=tokenizer,
        )
    else:
        baseline_metric = None

    # sample the inputs indices for the intervention
    all_indices = np.arange(len(dataset))
    input_indices = np.random.choice(
        all_indices,
        size=(nb_inter, nb_variable_inter + 1),
        replace=True,  # indice 0 is the ref input
    )

    # run interchange on the model
    batch_logits = torch.tensor([])
    if alignement.hook_type == "transformerlens":
        batch_logits = tl_apply_hooks(
            model,
            dataset_tok,
            variables_inter,
            alignement,
            input_indices,
            batch_size,
            nb_inter,
            verbose,
        )
    elif alignement.hook_type == "hf":
        batch_logits = hf_apply_hooks(
            model,
            dataset_tok,
            variables_inter,
            alignement,
            input_indices,
            batch_size,
            nb_inter,
            verbose,
        )

    # run interchange the high-level causal model
    causal_graph_output = []
    for i in range(nb_inter):
        ref_input = dataset[input_indices[i, 0]].causal_graph_input
        fixed_inputs = {
            variable: dataset[input_indices[i, k + 1]].causal_graph_input
            for k, variable in enumerate(variables_inter)
        }
        causal_graph_output.append(
            causal_graph.run(inputs=ref_input, fixed_inputs=fixed_inputs)
        )

    # check that the output of the model and the causal graph are the same
    interchange_metric = compute_metric(
        tokenizer,
        causal_graph_output,
        batch_logits,
        dataset,
        [int(x) for x in input_indices[:, 0]],
    )

    if baseline_metric is None:
        return interchange_metric
    else:
        return baseline_metric, interchange_metric


def InterchangeInterventionAccuracy(
    tokenizer: Any,
    causal_graph_output: List[Any],
    batch_logits: torch.Tensor,
    dataset: Dataset,
    target_idx: List[int],
    position: WildPosition,
    compute_mean: bool = True,
    verbose: bool = False,
):
    assert len(batch_logits.shape) == 3
    end_logits = batch_logits[
        range(len(target_idx)), position.positions_from_idx(target_idx), :
    ]
    # get the predicted token
    predicted_token = torch.argmax(end_logits, dim=-1)
    # detokenize
    predicted_str = tokenizer.batch_decode(predicted_token)  # type: ignore
    # compute the accuracy

    if verbose:  # TODO maybe remove
        for i in range(len(dataset)):
            print(f"{predicted_str[i]}, {causal_graph_output[i]}")

    results = [
        predicted_str[i] == causal_graph_output[i] for i in range(len(target_idx))
    ]
    if compute_mean:
        return np.mean(results)
    else:
        return results


# %%
