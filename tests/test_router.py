from itertools import combinations

import pytest

from nexussim.network import BaseNetwork
from nexussim.program import Program, sequence
from nexussim.router import Router
from nexussim.segments.behaviour import ConstantLoadSegment


# Basic test for routing between two software components
def test_router_basic():
    allocations = {"sw1": "c1", "sw2": "c2"}
    net1 = BaseNetwork(max_bandwidth=50, max_latency=0.5)
    net2 = BaseNetwork(max_bandwidth=100, max_latency=0.5)
    net3 = BaseNetwork(max_bandwidth=200, max_latency=1.0)
    net1.connect("c1")
    net1.connect("c2")
    net2.connect("c1")
    net2.connect("c2")
    net3.connect("c1")
    net3.connect("c2")
    networks = {
        "net1": net1,
        "net2": net2,
        "net3": net3,
    }
    software_components = {
        "sw1": Program(name="sw1", body=ConstantLoadSegment(duration=10)),
        "sw2": Program(name="sw2", body=ConstantLoadSegment(duration=10)),
    }
    router = Router(allocations, networks, software_components)
    # Should pick net2 (lowest latency and highest bandwidth among ties)
    assert router.get_network("sw1", "sw2") == "net2"
    assert router.get_network("sw2", "sw1") == "net2"


# Test with nested segments
def test_router_nested_segments():
    allocations = {"sw1": "c1", "sw2": "c2"}
    net1 = BaseNetwork(max_bandwidth=100, max_latency=1.0)
    net1.connect("c1")
    net1.connect("c2")
    networks = {
        "net1": net1,
    }

    segA = ConstantLoadSegment(duration=10)
    segA.suffix("segA")
    segB = ConstantLoadSegment(duration=10)
    segB.suffix("segB")

    seq = sequence(segA, segB)
    seq.suffix("prog1")
    prog1 = Program(name="sw1", body=seq)

    seg = ConstantLoadSegment(duration=10)
    seg.suffix("seg")
    prog2 = Program(name="sw2", body=seg)
    # assert False, f"Program body: {p.body}, segments: {p.body.segments}, {[s.id for s in p.body.segments]}"
    software_components = {
        "sw1": prog1,
        "sw2": prog2,
    }
    router = Router(allocations, networks, software_components)
    print(router)
    # All segments should be routable via net1
    # Check all combinations of segments and software components to ensure they are all routable via net1
    combos = list(combinations(("sw1", "sw2", "sw1::prog1", "sw1::prog1::segB", "sw1::prog1::segA", "sw2::seg"), 2))
    assert len(combos) == 15
    for c1, c2 in combos:
        assert router.get_network(c1, c2) == router.get_network(c2, c1) == "net1"


# Test no common network
def test_router_no_common_network():
    allocations = {"sw1": "c1", "sw2": "c2"}
    net1 = BaseNetwork(max_bandwidth=100, max_latency=1.0)
    net1.connect("c1")
    net2 = BaseNetwork(max_bandwidth=100, max_latency=1.0)
    net2.connect("c2")
    networks = {
        "net1": net1,
        "net2": net2,
    }
    software_components = {
        "sw1": Program(name="sw1", body=ConstantLoadSegment(duration=10)),
        "sw2": Program(name="sw2", body=ConstantLoadSegment(duration=10)),
    }
    router = Router(allocations, networks, software_components)
    assert router.get_network("sw1", "sw2") is None
    assert router.get_network("sw2", "sw1") is None


# Test symmetric key
def test_router_key_symmetry():
    allocations = {"a": "c1", "b": "c2"}
    net1 = BaseNetwork(max_bandwidth=100, max_latency=1.0)
    net1.connect("c1")
    net1.connect("c2")
    networks = {"net1": net1}
    software_components = {
        "a": Program(name="a", body=ConstantLoadSegment(duration=10)),
        "b": Program(name="b", body=ConstantLoadSegment(duration=10)),
    }
    router = Router(allocations, networks, software_components)
    assert router.get_network("a", "b") == router.get_network("b", "a")


def test_router_invalid_segment_structure():
    allocations = {"a": "c1", "b": "c2"}
    net1 = BaseNetwork(max_bandwidth=100, max_latency=1.0)
    net1.connect("c1")
    net1.connect("c2")
    networks = {"net1": net1}
    progA = ConstantLoadSegment(duration=10)
    progB = Program(name="a", body=ConstantLoadSegment(duration=10))
    software_components = {"a": progA, "b": progB}
    with pytest.raises(
        ValueError, match="Software component a must have a 'body' attribute representing its main segment."
    ):
        Router(allocations, networks, software_components)
