from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
import uuid

a = 0

NO_FUNCTION = "NoFunction"
NoFunction = Literal["NoFunction"]


@define
class CausalGraph:
    name: str = field()
    output_type: type = field()
    f: Callable[..., Any] | NoFunction = field(default=NO_FUNCTION)
    children: List["CausalGraph"] = field(factory=list)
    leaf: bool = field(default=False)
    uuid: str = field(init=False)

    def __attrs_post_init__(self):
        if self.leaf:
            assert self.f == NO_FUNCTION, "Leaf nodes should not have a function"
            assert len(self.children) == 0, "Leaf nodes should not have children"
        else:
            assert self.f != NO_FUNCTION, "Non-leaf nodes should have a function"
            assert len(self.children) > 0, "Non-leaf nodes should have children"
        self.uuid = str(uuid.uuid4())
        self.check_name_unique()

    def run(
        self,
        inputs: Dict[str, Any],
        fixed_vals: Dict[str, Any] = {},
        fixed_inputs: Dict[str, Dict[str, Any]] = {},
    ) -> Any:
        """Run the causal graph with the given inputs (for leaf nodes) and fixed values (for intermediate nodes). The values of intermediate nodes can also be fixed by passing a dictionary fixed_inputs, forcing them to take the values they have on the fixed input corresponding to their name."""
        if self.name in fixed_inputs:  # fixed_input is highest priority
            output = self.run(
                inputs=fixed_inputs[
                    self.name
                ],  # we don't propagate fixed_inputs and fixed_vals
            )
        elif self.leaf:
            if not self.name in inputs:
                raise ValueError(f"{self.name} not in inputs despite being a leaf")
            output = inputs[self.name]
        elif self.name in fixed_vals:
            output = fixed_vals[self.name]
        else:
            children_outputs = {
                c.name: c.run(inputs, fixed_vals, fixed_inputs) for c in self.children
            }
            assert self.f != NO_FUNCTION, "Non-leaf nodes should have a function"
            output = self.f(**children_outputs)

        if not isinstance(output, self.output_type):
            raise TypeError(
                f"Output {output} is not of type {self.output_type}"
            )  # mannualy checking the type
        return output

    def check_name_unique(self, names: Optional[Dict[str, str]] = None) -> None:
        """Check that the name of the causal graph is unique among the full graph"""
        if names is None:
            names = {}  # need to define a new set each time
        if self.name in names:
            if names[self.name] != self.uuid:
                raise ValueError(f"Name {self.name} is not unique")
        names[self.name] = self.uuid
        for child in self.children:
            child.check_name_unique(names)

    def get_all_nodes_names(self) -> Set[str]:
        """Get all nodes in the causal graph"""
        nodes = set([self.name])
        for child in self.children:
            nodes = nodes.union(child.get_all_nodes_names())
        return nodes
