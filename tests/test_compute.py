import pytest

from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.network import BaseNetwork
from nexussim.segments.base import Segment, SegmentState


class DummySegment(Segment):
    def _segment_body(self, context):
        yield self._schedule_after(10, self.RESUME_SEGMENT)


@pytest.fixture
def compute():
    return Compute(
        cpu=UnboundedCPU(10, cpu_speed=1000),
        memory=UnboundedMemory(1000),
        networks={"lo": BaseNetwork()},
    )


@pytest.fixture
def compute_no_network():
    return Compute(
        cpu=UnboundedCPU(10, cpu_speed=1000),
        memory=UnboundedMemory(1000),
    )


def test_compute_execute(compute):
    assert list(compute.networks) == ["lo"]
    segment = DummySegment()

    compute.execute(segment)
    assert segment.state == SegmentState.RUNNING


def test_compute_execute_invalid(compute):
    with pytest.raises(AttributeError):
        compute.execute("not a segment")


def test_compute_execute2(compute_no_network):
    assert list(compute_no_network.networks) == []
    segment = DummySegment()

    compute_no_network.execute(segment)
    assert segment.state == SegmentState.RUNNING


def test_compute_execute_invalid2(compute_no_network):
    with pytest.raises(AttributeError):
        compute_no_network.execute("not a segment")
