
# @define
# class WildPosition:
#     """Stores the position of a component, can either be a single int or an array of int (e.g. if each data point has correspond to a different position)."""

#     position: Union[int, List[int], torch.Tensor] = field()
#     label: str = field(kw_only=True)

#     def positions_from_idx(self, idx: List[int]) -> List[int]:
#         if isinstance(self.position, int):
#             return [self.position] * len(idx)
#         else:
#             assert max(idx) < len(
#                 self.position
#             ), f"Index out of range! {max(idx)} > {len(self.position)}"
#             return [int(self.position[idx[i]]) for i in range(len(idx))]

#     def __attrs_post_init__(self):
#         if isinstance(self.position, torch.Tensor):
#             assert self.position.dim() == 1
#             self.position = [int(x) for x in self.position.tolist()]


# @define
# class ModelComponent:
#     """Stores a model component (head, layer, etc.) and its position in the model.
#     * q, k, v, z refers to individual heads (head should be specified).
#     * resid, mlp, attn refers to the whole layer (head should not be specified)."""

#     position: WildPosition = field(kw_only=True, init=False)
#     layer: int = field(kw_only=True)
#     name: str = field(kw_only=True)
#     head: int = field(factory=lambda: NOT_A_HEAD, kw_only=True)
#     hook_name: str = field(init=False)

#     @name.validator  # type: ignore
#     def check(self, attribute, value):
#         assert value in ["q", "k", "v", "z", "resid_pre", "resid_post", "mlp", "attn"]

#     def __init__(
#         self,
#         position: Union[int, torch.Tensor, WildPosition],
#         position_label: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         if not isinstance(position, WildPosition):
#             assert position_label is not None, "You should specify a position label!"
#             self.position = WildPosition(position=position, label=position_label)
#         else:
#             self.position = position

#         self.__attrs_init__(**kwargs)  # type: ignore

#     def __attrs_post_init__(self):
#         if self.name in ["q", "k", "v", "z"]:
#             self.hook_name = utils.get_act_name(self.name, self.layer, "a")
#             assert self.head != NOT_A_HEAD, "You should specify a head number!"
#         else:
#             assert self.head == NOT_A_HEAD, "You should not specify a head number!"
#             self.head = NOT_A_HEAD
#             if self.name in ["resid_pre", "resid_post"]:
#                 self.hook_name = utils.get_act_name(self.name, self.layer)
#             elif self.name == "mlp":
#                 self.hook_name = utils.get_act_name("mlp_out", self.layer)
#             elif self.name == "attn":
#                 self.hook_name = utils.get_act_name("attn_out", self.layer)

#         assert isinstance(self.position, WildPosition)

#     def is_head(self):
#         return self.head != NOT_A_HEAD

#     def __str__(self):
#         if self.is_head():
#             head_str = f".h{self.head}"
#         else:
#             head_str = ""
#         return f"{self.hook_name}{head_str}@{self.position.label}"

#     def __repr__(self):
#         return str(self)

#     def __hash__(self):
#         return hash(str(self))
