from causal_checker.causal_graph import CausalGraph, NoFunction
import pytest


def test_causal_graph():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)

    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])
    assert c.run({"a": 1, "b": 2}) == "3"


def test_fixing_values():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

    g = lambda c: c + "!"
    d = CausalGraph(name="d", output_type=str, f=g, children=[c])

    assert d.run({"a": 1, "b": 2}) == "3!"
    assert d.run({"a": 1, "b": 2}, fixed_vals={"c": "bla"}) == "bla!"


def test_fixing_inputs():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

    g = lambda c: c + "!"
    d = CausalGraph(name="d", output_type=str, f=g, children=[c])

    assert d.run({"a": 1, "b": 2}, fixed_inputs={"c": {"a": 5, "b": 10}}) == "15!"

    assert (
        d.run({"a": 1, "b": 2}, fixed_inputs={"a": {"a": 5, "b": 10}}) == "7!"
    ), "fixed_input is prioritized over inputs"
    assert (
        d.run(
            {"a": 1, "b": 2}, fixed_inputs={"c": {"a": 5, "b": 10}}, fixed_vals={"c": 8}
        )
        == "15!"
    ), "fixed_input is prioritized over fixed_vals"


def test_no_enough_inputs():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])
    with pytest.raises(ValueError):
        c.run({"b": 2})


def test_wrong_type():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

    g = lambda c: c + "!"
    d = CausalGraph(name="d", output_type=bool, f=g, children=[c])

    with pytest.raises(TypeError):
        d.run({"a": 1, "b": "2"})

    with pytest.raises(TypeError):
        c.run({"a": 1, "b": 2}, fixed_vals={"c": 3})

    with pytest.raises(TypeError):
        c.run({"a": 1, "b": "2"})


def test_causal_graph_name_unique():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

    c.check_name_unique()

    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    with pytest.raises(ValueError):
        f = lambda a, b: str(a + b)
        c = CausalGraph(name="a", output_type=str, f=f, children=[a, b])


def test_get_all_nodes_names():
    a = CausalGraph(name="a", output_type=int, leaf=True)
    b = CausalGraph(name="b", output_type=int, leaf=True)

    f = lambda a, b: str(a + b)
    c = CausalGraph(name="c", output_type=str, f=f, children=[a, b])

    assert c.get_all_nodes_names() == {"a", "b", "c"}
