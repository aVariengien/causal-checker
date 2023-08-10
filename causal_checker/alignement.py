# %%
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal, Union
from causal_checker.causal_graph import CausalGraph
from swap_graphs.core import ModelComponent, WildPosition, ActivationStore
from transformer_lens import HookedTransformer
import numpy as np
import random as rd
import torch
from causal_checker.retrieval import CausalInput, ContextQueryPrompt, OperationDataset
from causal_checker.hf_hooks import residual_steam_hook_fn
from causal_checker.utils import get_first_token_id
from causal_checker.retrieval import find_answer

Metric = Callable[
    [Any, List[Any], torch.Tensor, OperationDataset, List[int], Optional[Dict]],
    Any,
]


@define
class CausalAlignement:
    mapping_tl: Dict[str, List[ModelComponent]] = field(factory=dict)
    mapping_hf: Dict[str, Callable] = field(factory=dict)
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
    dataset: OperationDataset,
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
    dataset_str = [p.model_input for p in dataset]
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"]).cuda()  # type: ignore
    logits = batched_forward(model, batch_size, dataset_tok)
    cg_output = []
    for i in range(len(dataset)):
        cg_output.append(causal_graph.run(inputs=dataset[i].causal_graph_input))
    baseline_metric = compute_metric(
        tokenizer, cg_output, logits, dataset, list(range(len(dataset))), None
    )
    return baseline_metric


def check_alignement(
    alignement: CausalAlignement,
    model: HookedTransformer,
    causal_graph: CausalGraph,
    dataset: OperationDataset,
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
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"]).cuda()  # type: ignore

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
        {"input_indices": input_indices, "variables_inter": variables_inter},
    )

    if baseline_metric is None:
        return interchange_metric
    else:
        return baseline_metric, interchange_metric


def fn_eq(f1: Callable, f2: Callable) -> bool:
    return (
        f1.__code__.co_code == f2.__code__.co_code
        and f1.__closure__ == f2.__closure__
        and f1.__code__.co_consts == f2.__code__.co_consts
    )


def check_alignement_batched_graphs(
    alignements: List[CausalAlignement],
    model: HookedTransformer,
    causal_graphs: List[CausalGraph],
    dataset: OperationDataset,
    metrics: Dict[str, Metric],
    list_variables_inter: List[List[str]],
    nb_inter: int = 100,
    batch_size: int = 10,
    verbose: bool = False,
    tokenizer: Any = None,
    seed: Optional[int] = None,
) -> Any:
    """
    When the aligment of different high-level causal graphs lead to the same intervention on the low-level graph (NN),
    we can perform a single set of intervention on the low-level graph and compare the output to the different high-level graphs on the same intervention.
    Only support hunging face models for now.
    """
    np.random.seed(seed)
    assert len(causal_graphs) == len(list_variables_inter) == len(alignements)
    for alig in alignements:
        assert alig.hook_type == "hf"

    # check that the aligments lead to the same intervention on the low-level graph (NN)
    fns = [
        alignements[0].mapping_hf[node_name] for node_name in list_variables_inter[0]
    ]
    for i in range(len(alignements)):
        for j, node_name in enumerate(list_variables_inter[i]):
            if not fn_eq(alignements[i].mapping_hf[node_name], fns[j]):
                raise ValueError(
                    "Hook function of different alignements should be the same"
                )
    ref_alignement, ref_variables_inter = alignements[0], list_variables_inter[0]
    # since all alignements lead to the same intervention on the low-level graph (NN), we can use the first one as a reference

    nb_variable_inter = len(ref_variables_inter)

    assert (
        tokenizer is not None
    ), "tokenizer must be provided if model is not a HookedTransformer"

    dataset_str = [i.model_input for i in dataset]
    dataset_tok = torch.tensor(tokenizer(dataset_str, padding=True)["input_ids"]).cuda()  # type: ignore

    # sample the inputs indices for the intervention
    all_indices = np.arange(len(dataset))
    input_indices = np.random.choice(
        all_indices,
        size=(nb_inter, nb_variable_inter + 1),
        replace=True,  # indice 0 is the ref input
    )

    # run interchange on the model
    batch_logits = hf_apply_hooks(
        model,
        dataset_tok,
        ref_variables_inter,
        ref_alignement,
        input_indices,
        batch_size,
        nb_inter,
        verbose,
    )

    # run interchange the high-level causal model
    all_causal_graph_outputs = []

    for setup_idx in range(len(causal_graphs)):
        causal_graph_outputs = []
        for i in range(nb_inter):
            ref_input = dataset[input_indices[i, 0]].causal_graph_input
            fixed_inputs = {
                variable: dataset[input_indices[i, k + 1]].causal_graph_input
                for k, variable in enumerate(list_variables_inter[setup_idx])
            }
            causal_graph_outputs.append(
                causal_graphs[setup_idx].run(
                    inputs=ref_input, fixed_inputs=fixed_inputs
                )
            )
        all_causal_graph_outputs.append(causal_graph_outputs)

    # check that the output of the model and the causal graph are the same
    all_metrics = {metric_name: [] for metric_name in metrics.keys()}
    for metric_name, metric in metrics.items():
        for setup_idx, causal_graph_outputs in enumerate(all_causal_graph_outputs):
            interchange_metric = metric(
                tokenizer,
                causal_graph_outputs,
                batch_logits,
                dataset,
                [int(x) for x in input_indices[:, 0]],
                {
                    "input_indices": input_indices,
                    "variables_inter": list_variables_inter[setup_idx],
                },
            )
            all_metrics[metric_name].append(interchange_metric)

    return all_metrics


def is_prefix(s1: str, s2: str) -> bool:
    """Check if s1 is a prefix of s2"""
    if len(s1) > len(s2):
        return False
    return s1 == s2[: len(s1)]


def soft_match(s1: str, s2: str) -> bool:
    s1 = s1.lower().replace(" ", "")
    s2 = s2.lower().replace(" ", "")
    return is_prefix(s1, s2) or is_prefix(s2, s1)


def InterchangeInterventionAccuracy(
    tokenizer: Any,
    causal_graph_output: List[Any],
    batch_logits: torch.Tensor,
    dataset: OperationDataset,
    target_idx: List[int],
    extra_data: Optional[Dict] = None,
    compute_mean: bool = True,
    verbose: bool = False,
    soft_matching: bool = False,
):
    assert len(batch_logits.shape) == 3
    end_logits = batch_logits[
        range(len(target_idx)),
        dataset.get_end_position().positions_from_idx(target_idx),
        :,
    ]
    # get the predicted token
    predicted_token = torch.argmax(end_logits, dim=-1)
    # detokenize
    predicted_str = tokenizer.batch_decode(predicted_token)  # type: ignore
    # compute the accuracy

    if verbose:  # TODO maybe remove
        for i in range(len(dataset)):
            print(f"{predicted_str[i]}, {causal_graph_output[i]}")

    if not soft_matching:
        results = [
            predicted_str[i] == causal_graph_output[i] for i in range(len(target_idx))
        ]
    else:
        results = [
            soft_match(predicted_str[i], causal_graph_output[i])
            for i in range(len(target_idx))
        ]

    if compute_mean:
        return np.mean(results)
    else:
        return list(np.array(results, dtype=int))


def InterchangeInterventionTokenProbability(
    tokenizer: Any,
    causal_graph_output: List[Any],
    batch_logits: torch.Tensor,
    dataset: OperationDataset,
    target_idx: List[int],
    extra_data: Optional[Dict] = None,
    compute_mean: bool = True,
    verbose: bool = False,
):
    assert len(batch_logits.shape) == 3
    end_logits = batch_logits[
        range(len(target_idx)),
        dataset.get_end_position().positions_from_idx(target_idx),
        :,
    ]
    probas = torch.softmax(end_logits, dim=-1)

    corrected_tokens_id = [
        get_first_token_id(tokenizer, causal_graph_output[i])
        for i in range(len(target_idx))
    ]

    correct_probs = probas[range(len(target_idx)), corrected_tokens_id]

    if verbose:  # TODO maybe remove
        print(correct_probs.shape)
        for i in range(len(dataset)):
            print(f"{causal_graph_output[i]}, {correct_probs[i]}")

    if compute_mean:
        return torch.mean(correct_probs).float().cpu().numpy()
    else:
        return correct_probs.float().cpu().numpy()


def InterchangeInterventionLogitDiff(
    tokenizer: Any,
    causal_graph_output: List[Any],
    batch_logits: torch.Tensor,
    dataset: OperationDataset,
    target_idx: List[int],
    extra_data: Optional[Dict] = None,
    compute_mean: bool = True,
    verbose: bool = False,
):
    """This metric is only intended to be used in the context of the two-step retrival hypothesis"""
    assert len(batch_logits.shape) == 3

    input_indices = np.zeros((len(target_idx), 2), dtype=int)
    input_indices[:, 0] = np.array(target_idx)
    input_indices[:, 1] = np.array(target_idx)
    if extra_data is not None:
        assert extra_data["variables_inter"] in [
            ["nil"],
            ["query"],
            ["output"],
        ], "extra_data must be None or a dict with key 'variables_inter' in [['nil'], ['query'], ['output']]"
        if extra_data["variables_inter"] == ["output"]:
            input_indices = extra_data["input_indices"]

    end_logits = batch_logits[
        range(len(target_idx)),
        dataset.get_end_position().positions_from_idx(target_idx),
        :,
    ]
    corrected_tokens_id = [
        get_first_token_id(tokenizer, causal_graph_output[i])
        for i in range(len(target_idx))
    ]
    correct_logits = end_logits[range(len(corrected_tokens_id)), corrected_tokens_id]

    queries = [dataset.operations[idx].query for idx in target_idx]

    alt_tokens = []
    for i in range(len(target_idx)):
        rd_query = queries[rd.randint(0, len(queries) - 1)]
        tok = find_answer(rd_query, dataset.operations[input_indices[i, 1]].context)
        tries = 0
        while tok == causal_graph_output[i]:
            rd_query = queries[rd.randint(0, len(queries) - 1)]
            tok = find_answer(rd_query, dataset.operations[input_indices[i, 1]].context)
            tries += 1
            if tries > 50:
                print("WARNING: could not find an alternative token")
                break

        alt_tokens.append(tok)

    alt_ids = [
        get_first_token_id(tokenizer, alt_tokens[i]) for i in range(len(target_idx))
    ]
    alt_logits = end_logits[range(len(alt_ids)), alt_ids]

    logit_diff = correct_logits - alt_logits
    if verbose:  # TODO maybe remove
        print(logit_diff.shape)
        for i in range(len(dataset)):
            print(f"{causal_graph_output[i]}, {logit_diff[i]}")

    if compute_mean:
        return torch.mean(logit_diff).float().cpu().numpy()
    else:
        return logit_diff.float().cpu().numpy()


# %%

# %%

# %%

# %%

# %%

# %%
